# HGEN
The code of ICDM'21 paper "Deep Generation of Heterogeneous Networks"

The code has been tested under Python 3.7 and Pytorch 1.7.1

The data can be accessed [here](https://www.dropbox.com/sh/lmryy7r4la3owgj/AAB2eKhCp3UUEP5Nb8foJxUla?dl=0).  

For the real-world heterogneous graph inference:
```
python hgen.py
```

For the synthetic heterogneous graph inference:
```
python hgen_syn.py
```

All the hyperparameters are tuned and dataset have been attached. 
Run the following code to see available parameters that can be passed in:

```
python stgen.py -h
```

The following is the dependency of the project:
python=3.8.1  
pytorch=1.10  
networkx=2.5  
numpy=1.20.2  
