#p14
# metric per class

# Define precision metric
metric_precision = Precision(
    task="multiclass", num_classes=7, average=None)

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
precision = metric_precision.compute()

# Get precision per class
precision_per_class = {
    k: precision[v].item()
    for k, v 
    in dataset_test.class_to_idx.items()
}
print(precision_per_class)

''' {'cirriform clouds': 0.699999988079071, 'clear sky': 0.9384615421295166, 'cumulonimbus clouds': 0.800000011920929, 'cumulus clouds': 0.5819672346115112, 'high cumuliform clouds': 0.474683552980423, 'stratiform clouds': 0.7755101919174194, 'stratocumulus clouds': 0.761904776096344}'''
