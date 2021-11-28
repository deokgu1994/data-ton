import torch.nn.functional as F
import torch
import torch.nn as nn

# stem
class StemBlock(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # imge 512*512,3 -> 256*256, 63
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class StreamGenerateBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class StageBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, inputs):
        identity = inputs
        out = self.block(inputs)
        out += identity
        return self.relu(out)


# stage01
class Stage01StreamGenerateBlock(nn.Module):
    def __init__(self, W):  # W = default
        super().__init__()
        
        self.high_res_block = nn.Sequential(
            nn.Conv2d(256, W, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(W),
            nn.ReLU(),
        )

        self.medium_res_block = nn.Sequential(
            nn.Conv2d(256, W * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 2),
            nn.ReLU(),
        )

    def forward(self, inputs):
        out_high = self.high_res_block(inputs)
        out_medium = self.medium_res_block(inputs)
        return out_high, out_medium


class Stage01Block(nn.Module):
    def __init__(self, in_channels, W=48):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )

        if in_channels == 64:
            self.identity_block = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
            )
        self.relu = nn.ReLU()
        self.in_channels = in_channels

        self.stage01streamgenerateblock = Stage01StreamGenerateBlock(W)

    def forward(self, inputs) -> "return out_high, out_medium":
        identity = inputs
        out = self.block(inputs)

        if self.in_channels == 64:
            identity = self.identity_block(identity)

        out = self.relu(out + identity)

        return self.stage01streamgenerateblock(out)


# stage02
class Stage02Fuse(nn.Module):
    def __init__(self, W):
        super().__init__()

        self.high_to_medium = nn.Sequential(
            nn.Conv2d(W, W * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 2),
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(W * 2, W, kernel_size=1, bias=False), nn.BatchNorm2d(W)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])

        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        med2high = self.medium_to_high(med2high)

        high2med = self.high_to_medium(inputs_high)

        out_high = inputs_high + med2high
        out_medium = inputs_medium + high2med

        out_high = self.relu(inputs_high + med2high)
        out_medium = self.relu(inputs_medium + high2med)
        return out_high, out_medium


class Stage02(nn.Module):
    def __init__(self, W):
        super().__init__()

        high_res_blocks = [StageBlock(W) for _ in range(4)]
        medium_res_blocks = [StageBlock(W * 2) for _ in range(4)]

        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)

        self.stage02fuse = Stage02Fuse(W)

        self.midium_2_low = StreamGenerateBlock(W * 2)

    def forward(self, inputs_high, inputs_medium):
        out_high = self.high_res_blocks(inputs_high)

        out_medium = self.medium_res_blocks(inputs_medium)

        out_high, out_medium = self.stage02fuse(out_high, out_medium)

        out_low = self.midium_2_low(out_medium)

        return out_high, out_medium, out_low


# stage 03
class Stage03Fuse(nn.Module):
    def __init__(self, W):
        super().__init__()

        self.high_to_medium = nn.Sequential(
            nn.Conv2d(W, W * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 2),
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(W, W, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W),
            nn.ReLU(),
            nn.Conv2d(W, W * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 4),
        )
        self.medium_to_low = nn.Sequential(
            nn.Conv2d(W * 2, W * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 4),
        )

        self.medium_to_high = nn.Sequential(
            nn.Conv2d(W * 2, W, kernel_size=1, bias=False),
            nn.BatchNorm2d(W),
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(W * 4, W, kernel_size=1, bias=False), nn.BatchNorm2d(W)
        )
        self.low_to_medium = nn.Sequential(
            nn.Conv2d(W * 4, W * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(W * 2),
        )
        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium, inputs_low):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        medium_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])

        low2high = F.interpolate(
            inputs_low,
            size=high_size,
            mode="bilinear",
            align_corners=True,
        )
        low2high = self.low_to_high(low2high)

        low2med = F.interpolate(
            inputs_low,
            size=medium_size,
            mode="bilinear",
            align_corners=True,
        )
        low2med = self.low_to_medium(low2med)

        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        med2high = self.medium_to_high(med2high)

        high2low = self.high_to_low(inputs_high)
        high2med = self.high_to_medium(inputs_high)

        med2low = self.medium_to_low(inputs_medium)

        out_high = self.relu(inputs_high + med2high + low2high)
        out_medium = self.relu(inputs_medium + high2med + low2med)
        out_low = self.relu(inputs_low + high2low + med2low)
        return out_high, out_medium, out_low


class Stage03(nn.Module):
    def __init__(self, W):
        super().__init__()
        high_res_block = [StageBlock(W) for _ in range(4)]
        medium_res_block = [StageBlock(W * 2) for _ in range(4)]
        low_res_block = [StageBlock(W * 4) for _ in range(4)]

        self.high_res_blocks = nn.Sequential(*high_res_block)
        self.medium_res_blocks = nn.Sequential(*medium_res_block)
        self.low_res_blocks = nn.Sequential(*low_res_block)

        self.stage03Fuse = Stage03Fuse(W)
        self.low_to_vlow = StreamGenerateBlock(W * 4)

    def forward(self, inputs_high, inputs_medium, inputs_low):
        for _ in range(4):
            high_res_blocks = self.high_res_blocks(inputs_high)
            medium_res_blocks = self.medium_res_blocks(inputs_medium)
            low_res_block = self.low_res_blocks(inputs_low)

            inputs_high, inputs_medium, inputs_low = self.stage03Fuse(
                high_res_blocks, medium_res_blocks, low_res_block
            )
        out_vlow = self.low_to_vlow(inputs_low)
        return inputs_high, inputs_medium, inputs_low, out_vlow


# stage 04
class Stage04Fuse(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(W, W * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 2),
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(W, W, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W),
            nn.ReLU(),
            nn.Conv2d(W, W * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 4),
        )
        self.high_to_vlow = nn.Sequential(
            nn.Conv2d(W, W, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W),
            nn.ReLU(),
            nn.Conv2d(W, W, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W),
            nn.ReLU(),
            nn.Conv2d(W, W * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 8),
        )
        self.medium_to_low = nn.Sequential(
            nn.Conv2d(W * 2, W * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 4),
        )
        self.medium_to_vlow = nn.Sequential(
            nn.Conv2d(W * 2, W * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 2),
            nn.ReLU(),
            nn.Conv2d(W * 2, W * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 8),
        )
        self.low_to_vlow = nn.Sequential(
            nn.Conv2d(W * 4, W * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(W * 8),
        )

        # up
        self.vlow_to_low = nn.Sequential(
            nn.Conv2d(W * 8, W * 4, kernel_size=1, bias=False), nn.BatchNorm2d(W * 4)
        )
        self.vlow_to_medium = nn.Sequential(
            nn.Conv2d(W * 8, W * 2, kernel_size=1, bias=False), nn.BatchNorm2d(W * 2)
        )
        self.vlow_to_high = nn.Sequential(
            nn.Conv2d(W * 8, W, kernel_size=1, bias=False), nn.BatchNorm2d(W)
        )
        self.low_to_medium = nn.Sequential(
            nn.Conv2d(W * 4, W * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(W * 2),
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(W * 4, W, kernel_size=1, bias=False), nn.BatchNorm2d(W)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(W * 2, W, kernel_size=1, bias=False),
            nn.BatchNorm2d(W),
        )

        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium, inputs_low, inputs_vlow):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        medium_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])
        low_size = (inputs_low.shape[-1], inputs_low.shape[-2])

        # high
        vlow2high = F.interpolate(
            inputs_vlow,
            size=high_size,
            mode="bilinear",
            align_corners=True,
        )
        vlow2high = self.vlow_to_high(vlow2high)
        low2high = F.interpolate(
            inputs_low,
            size=high_size,
            mode="bilinear",
            align_corners=True,
        )
        low2high = self.low_to_high(low2high)
        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        med2high = self.medium_to_high(med2high)
        out_high = self.relu(inputs_high + med2high + low2high + vlow2high)

        # med
        vlow2med = F.interpolate(
            inputs_vlow, size=medium_size, mode="bilinear", align_corners=True
        )
        vlow2med = self.vlow_to_medium(vlow2med)
        low2med = F.interpolate(
            inputs_low,
            size=medium_size,
            mode="bilinear",
            align_corners=True,
        )
        low2med = self.low_to_medium(low2med)
        high2med = self.high_to_medium(inputs_high)
        out_medium = self.relu(high2med + inputs_medium + low2med + vlow2med)

        # low
        vlow2low = F.interpolate(
            inputs_vlow, size=low_size, mode="bilinear", align_corners=True
        )
        vlow2low = self.vlow_to_low(vlow2low)
        high2low = self.high_to_low(inputs_high)
        med2low = self.medium_to_low(inputs_medium)
        out_low = self.relu(high2low + med2low + inputs_low + vlow2low)

        # vlow
        high2vlow = self.high_to_vlow(inputs_high)
        med2vlow = self.medium_to_vlow(inputs_medium)
        low2vlow = self.low_to_vlow(inputs_low)
        out_vlow = self.relu(high2vlow + med2vlow + low2vlow + inputs_vlow)

        return out_high, out_medium, out_low, out_vlow


class Stage04(nn.Module):
    def __init__(self, W):
        super().__init__()

        high_res_block = [StageBlock(W) for _ in range(4)]
        medium_res_block = [StageBlock(W * 2) for _ in range(4)]
        low_res_block = [StageBlock(W * 4) for _ in range(4)]
        vlow_res_block = [StageBlock(W * 8) for _ in range(4)]

        self.high_res_blocks = nn.Sequential(*high_res_block)
        self.medium_res_blocks = nn.Sequential(*medium_res_block)
        self.low_res_blocks = nn.Sequential(*low_res_block)
        self.vlow_res_blocks = nn.Sequential(*vlow_res_block)

        self.stage04fuse = Stage04Fuse(W)

    def forward(self, inputs_high, inputs_medium, inputs_low, inputs_vlow):
        for _ in range(3):
            high_res_blocks = self.high_res_blocks(inputs_high)
            medium_res_blocks = self.medium_res_blocks(inputs_medium)
            low_res_block = self.low_res_blocks(inputs_low)
            vlow_res_block = self.vlow_res_blocks(inputs_vlow)

            inputs_high, inputs_medium, inputs_low, inputs_vlow = self.stage04fuse(
                high_res_blocks, medium_res_blocks, low_res_block, vlow_res_block
            )
        return inputs_high, inputs_medium, inputs_low, inputs_vlow


# last
class LastBlock(nn.Module):
    def __init__(self, W, num_classes):
        super().__init__()
        total_channels = W + W * 2 + W * 4 + W * 8
        self.block = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(),
            nn.Conv2d(total_channels, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, inputs_high, inputs_medium, inputs_low, inputs_vlow):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        original_size = (high_size[0] * 4, high_size[1] * 4)

        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        low2high = F.interpolate(
            inputs_low, size=high_size, mode="bilinear", align_corners=True
        )
        vlow2high = F.interpolate(
            inputs_vlow, size=high_size, mode="bilinear", align_corners=True
        )

        out = torch.cat([inputs_high, med2high, low2high, vlow2high], dim=1)
        out = self.block(out)

        out = F.interpolate(
            out, size=original_size, mode="bilinear", align_corners=True
        )
        return out


class HRNetV2(nn.Module):
    def __init__(self, num_classes=11, W=48, supervision=False):
        super().__init__()
        self.stemblock = StemBlock()
        self.stage01block = Stage01Block(64, W)
        self.stage02 = Stage02(W)
        self.stage03 = Stage03(W)
        self.stage04 = Stage04(W)
        self.lastblock = LastBlock(W, num_classes)

    def forward(self, input):
        stem = self.stemblock(input)

        out_high, out_medium = self.stage01block(stem)

        out_high, out_medium, out_low = self.stage02(out_high, out_medium)

        out_high, out_medium, out_low, out_vlow = self.stage03(
            out_high, out_medium, out_low
        )

        out_high, out_medium, out_low, out_vlow = self.stage04(
            out_high, out_medium, out_low, out_vlow
        )

        out = self.lastblock(out_high, out_medium, out_low, out_vlow)
        # print(f"out.shap {out.shape}")
        return out


if __name__ == "__main__":
    model = HRNetV2(num_classes=11, W=48, supervision=False)
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())
