# SOR

Same Object Retrieval project for Computer Vision @ Sapienza/Computer-Science M.Sc

Reference Paper: [An Improved Faster R-CNN for Same Object Retrieval](https://ieeexplore.ieee.org/document/7986979)

## [Slides](https://docs.google.com/presentation/d/131dm328uOeMdXZVKe87L_cg9ZjvruuswsrjVJDqtC2E/edit?usp=sharing)

## Instructions to run

* Clone the repository and create the following directories
    ```bash
    git clone https://github.com/Projectdotpy/SOR.git
    cd SOR
    mkdir data             # data folder containing the dataset
    mkdir model            # model folder containing the weights of the nn
    mkdir dist/query       # will contain the imgs given as query by the client
    cd dist
    ln -s ../data data     # needed to display the result in the browser
    cd ..
    ```

* Download `instre_monuments.zip` [TODO: add link] which contains the subset
  of images from the INSTRE dataset having only monuments and move it under the
  `data` folder

* Run `complex_data_transformation.sh` which will also run `create_csv.py`

    ```bash
    ./complex_data_transformation.sh
    ```

* Train the Network with the notebook present at [TODO: add link].
  This step will create some files in drive namely `model_frcnn_vgg.hdf5`,
  `record.csv` and `model_vgg_config.pickle`

* Place the `model_frcnn_vgg.hdf5` and `record.csv` files under `model` 
  and `model_vgg_config.pickle` under `data/instre_monuments`
  
* Hack: create `SOR` directory to be compliant with the packaging
  of the notebook and copy `faster_rcnn.py` for pickle deserialization
  ```bash
  mkdir SOR
  cp faster_rcnn.py SOR
  ```

* Modify the `model_vgg_config.pickle` file as follows
  ```python
    import pickle
    from faster_rcnn import Config
    with open('data/instre_monuments/model_vgg_config.pickle', 'rb') as f:
        C = pickle.load(f)
    C.model_path = 'model/model_frcnn_vgg.hdf5'
    with open('data/instre_monuments/model_vgg_config.pickle', 'wb') as f:
        pickle.dump(C, f)
  ```

* Run `create_retrieval_db.py`. This may take a while
  ```bash
  ./create_retrieval_db.py # generates two files under retrieval_db
  ```

* Launch `server.py`
  ```bash
  ./server.py
  ```

* Browse to http://127.0.0.1:5000/index.html
