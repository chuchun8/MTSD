# MTSD

ACL 2021 (Findings) paper: A Multi-Task Learning Framework for Multi-Target Stance Detection.

## Abstract

Multi-target stance detection aims to identify the stance taken toward a pair of different targets from the same text, and typically, there are multiple target pairs per dataset. Existing works generally train one model for each target pair. However, they fail to learn target-specific representations and are prone to overfitting. In this paper, we propose a new training strategy under the multi-task learning setting by training one model on all target pairs, which helps the model learn more universal representations and alleviate overfitting. Moreover, in order to extract more accurate target-specific representations, we propose a multi-task learning network which can jointly train our model with a stance (dis)agreement detection task that is designed to identify agreement and disagreement between stances in paired texts. Experimental results demonstrate that our proposed model outperforms the best-performing baseline by 12.39% in macro-averaged F1-score.

## Run

BERTweet is used as our baseline for multi-target stance detection in this paper. First, configure the environment:
```
$ pip install -r requirements.txt
```
For BERTweet-A in `merged` training setting, run
```
cd src/
python train_model.py \
    --input_target all \
    --model_select BERTweet \
    --train_mode unified \
    --col Stance1 \
    --lr 2e-5 \
    --batch_size 32 \
    --epochs 20 \
    --dropout 0. \
    --alpha 0.5
```
For BERTweet-A in `adhoc` training setting and target `Hillary Clinton` of target pair `Trump-Clinton`, run
```
cd src/
python train_model.py \
    --input_target trump_hillary \
    --model_select BERTweet \
    --train_mode adhoc \
    --col Stance2 \
    --lr 2e-5 \
    --batch_size 32 \
    --epochs 20 \
    --dropout 0. \
    --alpha 0.4
```
`input_target` can take one of the following target-pairs [`trump_hillary`, `trump_ted`, `hillary_bernie`] in adhoc setting and take [`all`] in merged setting.

`model_select` includes two options: [`BERTweet` and `BERT`].

`col` indicates the target in each target-pair. For example, for the target-pair `Trump-Clinton`, we have `Stance1` for Trump and `Stance2` for Clinton.

## Contact Info

Please contact Yingjie Li at yli300@uic.edu with any questions.
