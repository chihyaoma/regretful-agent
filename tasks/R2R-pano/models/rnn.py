import torch
from torch import nn


class CustomRNN(nn.Module):
    """
    A module that runs multiple steps of RNN cell
    With this module, you can use mask for variable-length input
    """
    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(CustomRNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, mask, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_[time], hx=hx)
            mask_ = mask[time].unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask_ + hx[0]*(1 - mask_)
            c_next = c_next*mask_ + hx[1]*(1 - mask_)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, mask, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
            mask = mask.transpose(0, 1)
        max_time, batch_size, _ = input_.size()

        if hx is None:
            hx = input_.new(batch_size, self.hidden_size).zero_()
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = CustomRNN._forward_rnn(
                cell=cell, input_=input_, mask=mask, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)