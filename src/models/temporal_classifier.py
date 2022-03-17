import torch
import torch.nn as nn
from src.models.memory import Memory

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()

class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding,True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv( x )
            out = self.batchnorm( out )
            out = self.chomp( out )
            return self.non_lin( out )

class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2,
                 relu_type = 'relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_kernels = len( kernel_sizes )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx ,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr( self ,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)

        for k_idx ,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr( self ,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs//self.num_kernels) != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self ,'cbcr0_{}'.format(k_idx))
            outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self ,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False, radius=16.0, n_slot=112, head=8):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']

        dilation_size = 1
        padding = [(s - 1) * dilation_size for s in self.ksizes]
        self.layer1 = MultibranchTemporalBlock(num_inputs, num_channels[0], self.ksizes, stride=1, dilation=dilation_size, padding=padding, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.mem1 = Memory(radius=radius, n_slot=n_slot, n_head=head, dim=num_channels[0], diff_aud_vid=True)

        dilation_size = 2
        padding = [(s - 1) * dilation_size for s in self.ksizes]
        self.layer2 = MultibranchTemporalBlock(num_channels[0], num_channels[1], self.ksizes, stride=1, dilation=dilation_size, padding=padding, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.mem2 = Memory(radius=radius, n_slot=n_slot, n_head=head, dim=num_channels[0], diff_aud_vid=True)

        dilation_size = 4
        padding = [(s - 1) * dilation_size for s in self.ksizes]
        self.layer3 = MultibranchTemporalBlock(num_channels[1], num_channels[2], self.ksizes, stride=1, dilation=dilation_size, padding=padding, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.mem3 = Memory(radius=radius, n_slot=n_slot, n_head=head, dim=num_channels[0], diff_aud_vid=True)

        dilation_size = 8
        padding = [(s - 1) * dilation_size for s in self.ksizes]
        self.layer4 = MultibranchTemporalBlock(num_channels[2], num_channels[3], self.ksizes, stride=1, dilation=dilation_size, padding=padding, dropout=dropout, relu_type=relu_type, dwpw=dwpw)

    def forward(self, x, aud, infer=False, mode='tr'):
        # return self.network(x)
        if mode == 'tr':
            x = self.layer1(x)
            _, tr_f, r_1, c_1 = self.mem1(x, aud, inference=infer, cha_first=True)
            tr_f = self.layer2(tr_f)
            _, tr_f, r_2, c_2 = self.mem2(tr_f, aud, inference=infer, cha_first=True)
            tr_f = self.layer3(tr_f)
            _, tr_f, r_3, c_3 = self.mem3(tr_f, aud, inference=infer, cha_first=True)
            x = self.layer4(tr_f)
        else:
            x = self.layer1(x)
            te_f, _, r_1, c_1 = self.mem1(x, aud, inference=infer, cha_first=True)
            te_f = self.layer2(te_f)
            te_f, _, r_2, c_2 = self.mem2(te_f, aud, inference=infer, cha_first=True)
            te_f = self.layer3(te_f)
            te_f, _, r_3, c_3 = self.mem3(te_f, aud, inference=infer, cha_first=True)
            x = self.layer4(te_f)
        return x, r_1 + r_2 + r_3, c_1 + c_2 + c_3
# --------------------------------

class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False, radius=16.0, n_slot=112, head=8):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw, radius=radius, n_slot=n_slot, head=head)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, aud, infer=False, mode='tr'):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out, recon, const = self.mb_ms_tcn(xtrans, aud, infer, mode)
        out = out.mean(2)
        return self.tcn_output(out), recon, const

class Temp_classifier(nn.Module):
    def __init__(self, radius, n_slot, head):
        super().__init__()

        tcn_options = {"dropout": 0.2, "dwpw": False, "kernel_size": [3,5,7], "num_layers": 4, "width_mult": 1}

        tcn_class = MultiscaleMultibranchTCN
        self.tcn = tcn_class(input_size=512,
                             num_channels=[256 * len([3,5,7]) * 1] * 4,
                             num_classes=500,
                             tcn_options=tcn_options,
                             dropout=tcn_options['dropout'],
                             relu_type='prelu',
                             dwpw=tcn_options['dwpw'],
                             radius=radius,
                             n_slot=n_slot,
                             head=head
                             )

    def forward(self, x, aud, infer=False, mode='tr'):
        x, recon, const = self.tcn(x, aud, infer, mode)
        return x, recon, const
