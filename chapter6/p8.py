#p8
# Define loss function for both RPN and R-CNN models
# remember that the RPN component classifies whether a region contains an object and predicts the bounding box coordinates 
# for the proposed regions.
# The R-CNN component classifies the object into one of multiple classes while also predicting the final bounding box coordinates.

# Implement the RPN classification loss function
rpn_cls_criterion = nn.BCEWithLogitsLoss()

# Implement the RPN regression loss function
rpn_reg_criterion = nn.MSELoss()

# Implement the R-CNN classification Loss function
rcnn_cls_criterion = nn.CrossEntropyLoss()

# Implement the R-CNN regression loss function
rcnn_reg_criterion = nn.MSELoss()