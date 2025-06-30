class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 6, 3)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 128

        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, 64, [6, 6], [3, 3]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, [4, 4], [2, 2]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )
        
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape([-1, self.final_flat])
        return self.dense(hidden)
    
    