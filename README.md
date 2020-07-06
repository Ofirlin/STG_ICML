## Usage 

To install the STG feature selection library, git clone the repo and run `pip install .` inside the repo.

For off-the-shelf usage (mainly for the R package):

```{python}
from stg import STG 
X, y = read_data()
model = STG(task_type='classification',input_dim=X_train.shape[1], output_dim=2, hidden_dims=[60, 20], activation='tanh',
            optimizer='SGD', learning_rate=1e-2, batch_size=X_train.shape[0], sigma=0.5, lam=0.02, random_state=1)
model.fit(X_train, y_train, nr_epochs=5000, print_interval=1000)
print(model.get_gates(mode='prob'))
y_pred = model.predict(new_X)
```

For custom model:
```
from stg.layers import FeatureSelector 

TBA
```
