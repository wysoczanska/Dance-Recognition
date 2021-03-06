from torchvision import utils


def visualize_filter(layer, writer, epoch):
    weight = layer.weight.data.cpu()
    writer.add_image('Conv1', utils.make_grid(weight), epoch)

