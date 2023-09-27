__author__ = 'tylin'
import pdb
import json

from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
from .clipscore.clipscore import ClipScore


class COCOEvalCap:
    def __init__(self, path, annotation_key='img_id',
                 metric=["B", "M",  "C", "CLIP"], do_average=True):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        data = json.load(open(path))
        imgids = []
        self.res_for_clipscore = []
        self.ground_truth, self.prediction = {}, {}
        annotation_ids_dict = {} # check for uniqueness
        self.metric = metric
        self.do_average = do_average

        imgid = 0
        for item in data:
            appended_item = {}
            img_path = item['imgpath']
            img_id = img_path.split('/')[-1].split('.')[0]
            appended_item["imgpath"] = img_path
            if "ds_name" in item.keys():
                appended_item["ds_name"] = item['ds_name']
                appended_item["bbox"] = item['bbox']  # [x1, y1, x2, y2]
            else:
                appended_item["ds_name"] = "none"
                appended_item["bbox"] = [-1,-1,-1,-1]  # [x1, y1, x2, y2]
            appended_item["gt"] = item['gt']

            if annotation_key == "img_id":
                annotation_id = img_id
            elif annotation_key == "region_id":
                annotation_id = item['region_id']
            elif annotation_key == "enum":
                annotation_id = imgid
            imgids.append(annotation_id)
            if annotation_id not in annotation_ids_dict:
                annotation_ids_dict[annotation_id] = True
                self.res_for_clipscore.append(appended_item)

            self.ground_truth[annotation_id] = [{'caption': i} for i in item['gt']]
            if "generated" in item.keys():
                self.prediction[annotation_id] = [{'caption': i for i in item['generated']}]
            else:
                self.prediction[annotation_id] = [{'caption': i for i in item['caption']}]

            imgid += 1

        self.params = {'image_id': imgids}
        self.num_total_preds = len(imgids)

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.ground_truth[imgId]
            res[imgId] = self.prediction[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = []
        if "B" in self.metric:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if "M" in self.metric:
            scorers.append((Meteor(),"METEOR"))
        if "R" in self.metric:
            scorers.append((Rouge(), "ROUGE_L"))
        if "C" in self.metric:
            scorers.append((Cider(), "CIDEr"))
        if "S" in self.metric:
            scorers.append((Spice(), "SPICE"))
        if "CLIP" in self.metric:
            scorers.append((ClipScore(self.res_for_clipscore), ["CLIPScore", "RefCLIPScore"]))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc*100, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc*100))
            else:
                self.setEval(score*100, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score*100))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
