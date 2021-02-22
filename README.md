# SimFin

Take the change vectors and their labels to apply auto encoder-decoder for semantic representation and with the encoded vectors, we apply distance calculation from the train set and the test set.

## Order of Scenario
1. save_light.py (train auto-encoder)
    a. description: takes only the train set data and makes scaler, auto-encoder model. Saves the scaler, model, encoded_trainset into files.
    b. example:
    <pre><code> python3 ./SimFin/save_light.py -t preprocessed -s 3 </pre></code>
2. distance_calculate.py (encode test set + calculate distances to train set)
    a. description: takes a test project and calculates the distance of each test instance from the train set.
    b. example: <pre><code> python3 ./SimFin/distance_calculate.py -t preprocessed -p maven </pre></code>
3. sort_top_k.py (sort the distances that are calculated)
    a. description: takes the calculated distance and sorts them to ascending order.
    b. example: <pre><code> python3 ./SimFin/sort_top_k.py preprocessed ranger </pre></code>


## Miscellaneous
1. inspection.py
    a. description: draws a graph of how a test instance’s distance is distributed.
    b. example: <pre><code> python3 ./SimFin/inspection.py preprocessed/sentry 0 clean </pre></code>
2. preprocess.py
    a. description: preprocesses to remove duplications or removing instances that are the same as buggy in the clean pool.
    b. example: <pre><code> python3 ./SimFin/preprocess_data.py preprocessed </pre></code>
3. gv_ae.py
    a. description: the full version (including the hunk information) of the previous scenario of the combined SimFin model. Training and prediction
4. gv_ae_no_hunk.py
    a. description: the light version (excluding the hunk information) of the combined model.
5. light.py
    a. description: don’t write vectors on file.
