#this method produces the predictions for the binary classification
def binary_predict(image,model):

    ypred = model.predict(image)
    y_pred = (ypred > 0.5)

    if y_pred == True:
        print(' tumor')
    else:
        print("healthy")

    return ypred
