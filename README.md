# gaussian processes for data regression and sampling

This code showcases the different parameters used for gaussian process regression.

Following Machine Learning and Pattern Recognition (Bishop) I implemented GPR in python
along with a small widget using matplotlib to be able to see the effects of the hyperparameters.

![Alt text](Demo_1.png?raw=true "Demo_1")

Sampling from the computed distribution:

![Alt text](Demo_2.png?raw=true "Demo_2")

Using a different kernel (and effectively modelling the stock market):

![Alt text](Demo_3.png?raw=true "Demo_3")

## Requirements:
```
numpy
matplotlib
```

## Using gpr.py:
import, fit:
```
import gpr
gp = gpr.GP()
gp.fit(data_x, data_y)
```

predict:
```
gp.predict(pred_x)
```
