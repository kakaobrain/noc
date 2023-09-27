### To Obtain Image Caption Evaluation Results

#### 1. Environment Setup:

To run the evaluation script, you need to first download the Standford models via the following command:
```yaml
chmod +x ./get_stanford_models.sh
./get_stanford_models.sh
```
#### 2. Perform Evaluation:

To evaluate the model's generated result, you can run the following command:
```yaml
chmod +x ./evaluation.sh
./evaluation.sh
```

After running the command, you should get the following results.

<img src="https://github.com/yxuansu/MAGIC/blob/main/image_captioning/evaluation/evaluation_result.png" width="400" height="450">


To evaluate different files, you should change the path of the --result_file_path argument in the script.

**[Note]** We have rigorously tested the evaluation scripts in Ubuntu 16.04 system. However, you might encounter JAVA errors on MacOS systems

#### 3. License
To measure BLEU, Meteor, Rouge-L, CIDEr, and SPICE scores, we followed this [repository](https://github.com/tylin/coco-caption) under the terms of the [FreeBSD license](./LICENSE). 

For the CLIPScore, we referred the [official implementation](https://github.com/jmhessel/clipscore) under the terms of the [MIT license](./pycocoevalcap/clipscore/LICENSE).