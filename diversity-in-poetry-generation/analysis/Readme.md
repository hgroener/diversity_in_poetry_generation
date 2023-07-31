# Length
Derive graphics and different metrics regarding poem length in sample or in the training data. To process a sample, run
```
python length.py --sample_path path_to_processed_sample \
--quatrain_path path_to_annotated_training_data \
--lang language \
--out_path path_where_graphics_and_metrics_are_saved \
--out_name name_of_output_files
```
Use the argument *--training_data_only* to derive metrics regarding training data.

# Lexical Diversity
Derive different metrics to measure the lexical diversity in a sample or in the training data.
```
python lexical_div.py --sample_path path_to_processed_sample \
--quatrain_path path_to_annotated_training_data \
--lang language
```
Use the argument *--training_data_only* to derive metrics regarding training data.

# Extractive Memorization
Determine to what extent a sample can be regarded as a copy from the training data. 
```
python lexical_div.py --sample_path path_to_processed_sample \
--quatrain_path path_to_annotated_training_data \
--cutoff cutoff
```
Use *--num_to_test* if the number of quatrains in a sample should be limitee (e.g. for debugging purposes).

# Meter and Rhyme
Derive different plots showcasing the distribution of meters and rhymes in a sample or in the training data. To process a sample, run
```
python mater_rhyme.py --sample_path path_to_processed_sample \
--quatrain_path path_to_annotated_training_data \
--save_path path_where_graphics_and_metrics_are_saved
```
Use the argument *--training_data_only* to derive metrics and plots regarding training data.

# Semantic Similarity
For each quatrain in a given sample, determine the most similar quatrain in the training data. Run
```
python semantic_similarity.py --sample_path path_to_processed_sample \
--emb_path path_to_embedded_training_data \
--lang language \
--save_path path_where_graphics_and_metrics_are_saved \
--out_name name_of_output_files
```
Use the argument *--training_data_only* together with *quatrain_path* to initially embed a training data corpus.

# Topic Modeling

Train topic models for training data corpora:
```
python train_topic_model.py --quatrain_path path_to_processed_training_data \
--workers number_of_cpu_threads \
--lang language \
--save_pat save_trained_model_to_path \
--min_topics lower_bound_for_number_of_topics_to_be_determined \
--max_topics upper_bound_for_number_of_topics_to_be_determined \
--stepsize stepsize
```
Use *--limit* in order to reduce the number of quatrains to be processed (e.g. for debugging purposes).
Process samples, derive most relevant topics in generated and training data:
```
python topic_modeling.py --sample_path path_to_processed_sample \
--trained_model_path path_to_saved_topic_model \
--lang language \
--save_path path_where_graphics_and_metrics_are_saved
```
Use *--training_data_only* to derive topic distributions and graphics regarding the training data.

