
import torch

def cal_val(val_loader, model, device):
    val_acc = []
    val_losses = []
    val_loss = 0
    val_total = 0
    val_correct = 0
    for _, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += targets.size(0)
        val_correct += predicted.eq(targets).sum().item()
#         val_losses.append(val_loss)
        val_acc.append(val_correct / val_total)
    
    return val_acc[-1], val_loss / len(val_loader)


def elements_provider(l):
    my_iterator = iter(l)

    def getter():
        nonlocal my_iterator
        while True:

            try:
                return next(my_iterator)
            except StopIteration:
                pass            
            my_iterator = iter(l)

    return getter