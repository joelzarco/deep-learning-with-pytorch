#p13
# Multi-class evaluation
'''
Not averaging, and analyzing the results per class;
Micro-averaging, ignoring the classes and computing the metrics globally;
Macro-averaging, computing metrics per class and averaging them;
Weighted-averaging, just like macro but with the average weighted by class size.
'''
# Both Precision and Recall have been imported

# Define metrics
metric_precision = Precision(task="multiclass", num_classes=7, average="micro")
metric_recall = Recall(task="multiclass", num_classes=7, average="micro")

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()
print(f"Precision: {precision}")
print(f"Recall: {recall}")