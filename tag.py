import argparse
import os
import json
import sys
import torch
import transformers
import copy
from transformers import BertTokenizerFast
from transformers import BertModel
from transformers import AutoModelForTokenClassification
from functools import cmp_to_key
import logging
import re
import pickle
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"]="false"

#general_counter=0
#general_counter_all=0

# GLOBAL VARIABLES (CAN BE USED TO CONFIGURE)
SCRIPT_PATH=os.path.abspath(os.path.dirname(__file__))
MODEL_SENTENCE_START_ID=101
MODEL_SENTENCE_END_ID=102
BATCH_SIZE=32
LANGUAGE_IDENTIFICATIOR_BATCH_SIZE=8
MANUAL_DEVICES=[]
LABEL_LIST_FILE = os.path.abspath(os.path.dirname(__file__)) + "/models/label_list.txt"
LABEL_CLASSES_FILE= SCRIPT_PATH + "/models/labels_classifier.txt"
LABEL_ORDER_FILE= SCRIPT_PATH + "/models/labels_order.json"
MODELS_DIR= SCRIPT_PATH + "/models"
NN_FULLFORM_LIST_PATH = SCRIPT_PATH + "/nn.pickle"
BM_FULLFORM_LIST_PATH = SCRIPT_PATH + "/bm.pickle"
BOKMAL_LABEL="B"
NYNORSK_LABEL="N"
BOKMAL_LABEL_ID=1
NYNORSK_LABEL_ID=2
NN_FULLFORM_LIST=None
BM_FULLFORM_LIST=None
ID2LABEL=None
LABEL2ID=None
LABEL_ORDER=None
PUNCTUATION=set([4,5,6,7,8,26,31,34,36,52,69])
IGNORE_BERT_TAGS={"$punc$"}
SUBST_TAG=18
PROP_TAG=64
GEN_TAG=46
UKJENT_TAG=20
SECOND_PERSON_TAG=2
EQUAL_TAGS={":subst:":"subst",
            ":ukjent:": "ukjent",
            ":adj:":"adj",
            ":prep:":"prep",
            ":verb:":"verb",
            ":det:":"det",
            ":konj:":"konj",
            ":pron:":"pron",
            ":adv:":"adv",
            ":inf-merke:":"inf-merke",
            ":<anf>:":"<anf>",
            ":sbu:":"sbu",
            ":clb:":"clb",
            ":<komma>:":"<komma>",
            ":<strek>:":"<strek>",
            ":<parentes-beg>:":"<parentes-beg>",
            ":<parentes-slutt>:":"<parentes-slutt>",
            ":interj:":"interj",
            ":symb:":"symb"
            }

NN_TO_BM ={
        "høfleg":"høflig",
        "eint":"ent",
        "<ikkje-clb>": "<ikke-clb>",
        "<ordenstal>": "<ordenstall>",
        "<romartal>" : "<romertall>",
        "bu": "be",
        "<st-verb>": "<s-verb>"
        }

MAIN_TAG_LIST_NN=["$punc$", "1", "2", "3", "<anf>", "<komma>", "<parentes-beg>", "<parentes-slutt>", "<strek>", "adj", "adv", "det", "inf-merke", "interj", "konj", "prep", "pron", "sbu", "subst", "symb", "ukjent", "verb", "<adj>", "<adv>", "<dato>", "<ellipse>", "<kolon>", "<next_token>", "<ordenstal>", "<perf-part>", "<pres-part>", "<punkt>", "<romartal>", "<semi>", "<spm>", "<st-verb>", "<utrop>", "akk", "appell", "bu", "dem", "eint", "fem", "fl", "fork", "forst", "gen", "hum", "høfleg", "imp", "inf", "komp", "kvant", "m/f", "mask", "nom", "nøyt", "pass", "perf-part", "pers", "pos", "poss", "pres", "pret", "prop", "refl", "res", "sp", "sup", "symb", "ub", "ubøy", "ufl"]
MAIN_TAG_LIST_BM=None
INT_TOKENIZATION_DEVICE=-1

def get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices):
    # Find one more available GPU:
    found_one=False
    for dev_num in final_devices:
        if dev_num in three_model_devices or dev_num in two_model_devices:
            final_devices.append(dev_num)
            found_one=True
            break
    if not found_one:
        for dev_num in three_model_devices:
            if dev_num not in final_devices:
                final_devices.append(dev_num)
                found_one=True
                break
            if not found_one:
                for dev_num in two_model_devices:
                    if dev_num not in final_devices:
                        final_devices.append(dev_num)
                        found_one=True
                        break
                if not found_one:
                    for dev_num in one_model_devices:
                        if dev_num not in final_devices:
                            final_devices.append(dev_num)
                            found_one=True
                            break
    return final_devices, found_one

def compare_label(t1,t2):
    global LABEL_ORDER
    val1=-1
    val2=-1
    key1 = t1 + " " + t2
    key2 = t2 + " " + t1
    if key1 in LABEL_ORDER:
        val1=LABEL_ORDER[key1]
    if key2 in LABEL_ORDER:
        val2=LABEL_ORDER[key2]
    if val1>val2:
        return -1
    return 1

def load_models_and_config():
    global SEGMENTATION_TOKENIZER
    global CLASSIFICATION_TOKENIZER
    global TOKENIZETION_TOKENIZER
    global SEGMENTATION_DEVICE
    global CLASSIFICATION_DEVICE
    global TOKENIZATION_DEVICE
    global SEGMENTATION_MODEL
    global CLASSIFICATION_MODEL
    global TOKENIZATION_MODEL
    global ID2LABEL
    global LABEL2ID
    global MODEL_SENTENCE_START_ID
    global MODEL_SENTENCE_END_ID
    global MAX_LENGTH_WITHOUT_CLS
    global BATCH_SIZE
    global LANGUAGE_IDENTIFICATIOR_BATCH_SIZE

    global NN_FULLFORM_LIST_PATH
    global BM_FULLFORM_LIST_PATH

    global BOKMAL_LABEL
    global NYNORSK_LABEL

    global BOKMAL_LABEL_ID
    global NYNORSK_LABEL_ID

    global TOKENS_STARTING_WITH_HASH

    global PUNCTUATION

    global CLASS_TO_LABEL_NN
    global CLASS_TO_LABEL_BM

    global NN_FULLFORM_LIST
    global BM_FULLFORM_LIST

    global MAIN_TAG_LIST_NN
    global MAIN_TAG_LIST_BM

    global MODEL_SENTENCE_START_ID
    global MODEL_SENTENCE_END_ID
    global BATCH_SIZE
    global LANGUAGE_IDENTIFICATIOR_BATCH_SIZE
    global MANUAL_DEVICES

    global LABEL_LIST_FILE
    global LABEL_CLASSES_FILE
    global LABEL_ORDER_FILE

    global IGNORE_BERT_TAGS
    global EQUAL_TAGS
    global NN_TO_BM

    global MAIN_TAG_LIST_NN
    global MAIN_TAG_LIST_BM

    global BOKMAL_LABEL
    global NYNORSK_LABEL
    global BOKMAL_LABEL_ID
    global NYNORSK_LABEL_ID
    global PUNCTUATION
    global MODELS_DIR
    global MANUAL_DEVICES

    global LABEL_ORDER
    global INT_TOKENIZATION_DEVICE

    # Try to identify NVIDIA devices.
    # -1 for CPU
    # If already set manually, use tem
    if len(MANUAL_DEVICES)==3:
                INT_CLASSIFICATION_DEVICE=MANUAL_DEVICES[0]
                INT_TOKENIZATION_DEVICE=MANUAL_DEVICES[1]
                INT_SEGMENTATION_DEVICE=MANUAL_DEVICES[2]

    # Else check PYNVML and try to figure out which device to use
    else:
        try:
            # Process os.environ["CUDA_VISIBLE_DEVICES"]
            import pynvml
            #from pynvml import *
            CUDA_TO_NV={}
            splitted_device_list=[]

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                splitted_device_list=os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                num = 0
                for i in splitted_device_list:
                    try:
                        dev_id=int(i)
                        CUDA_TO_NV[num]=dev_id
                        num+=1
                    except:
                        pass
            else:
                CUDA_TO_NV=None

            INT_CLASSIFICATION_DEVICE=-1  # -1 for cpu or gpu id
            INT_TOKENIZATION_DEVICE=-1    # -1 for cpu or gpu id
            INT_SEGMENTATION_DEVICE=-1  # -1 for cpu or gpu id

            three_model_devices=[]
            two_model_devices=[]
            one_model_devices=[]

            not_used_devices=[]

            final_devices=[]

            gpu_scores=[]

            one_model_size=8000000000

            pynvml.nvmlInit()

            # Try to find three GPUs with enough memory
            for dev_num in range(int(torch.cuda.device_count())):
            #t = torch.cuda.get_device_properties(dev_num).total_memory
            #r = torch.cuda.memory_reserved(dev_num)
                if CUDA_TO_NV is not None:
                    h = pynvml.nvmlDeviceGetHandleByIndex(CUDA_TO_NV[dev_num])
                else:
                    h = pynvml.nvmlDeviceGetHandleByIndex(dev_num)

                info = pynvml.nvmlDeviceGetMemoryInfo(h)

                # If three models fit
                if info.free > one_model_size*3:
                    three_model_devices.append(dev_num)
                    if info.used< 470000000 :
                        not_used_devices.append(dev_num)

                # If two models fit
                elif info.free > one_model_size*2:
                    two_model_devices.append(dev_num)
                    if info.used< 470000000 :
                        not_used_devices.append(dev_num)

                # If one model fits
                elif info.free > one_model_size:
                    one_model_devices.append(dev_num)
                    if info.used< 470000000 :
                        not_used_devices.append(dev_num)

            if len(not_used_devices)>=3:
                final_devices=not_used_devices[0:3]

            elif len(not_used_devices)==2:
                final_devices.append(not_used_devices[0])
                final_devices.append(not_used_devices[1])
                final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)

            elif len(not_used_devices)==1:
                final_devices.append(not_used_devices[0])
                final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)
                final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)

            elif len(not_used_devices)==0:
                final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)
                final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)
                final_devices, found_one = get_one_gpu(final_devices, one_model_devices, two_model_devices, three_model_devices)

            if len(final_devices)==1:
                for dev_num in final_devices:
                    if dev_num in three_model_devices:
                        if len(final_devices)<3:
                            final_devices.append(dev_num)
                        if len(final_devices)<3:
                            final_devices.append(dev_num)
                        break
                    elif dev_num in two_model_devices:
                        if len(final_devices)<3:
                            final_devices.append(dev_num)
                        if len(final_devices)==3:
                            break
            elif len(final_devices)==2:
                for dev_num in final_devices:
                    if dev_num in three_model_devices:
                        if len(final_devices)<3:
                            final_devices.append(dev_num)
                        break
                    elif dev_num in two_model_devices:
                        if len(final_devices)<3:
                            final_devices.append(dev_num)
                        break

            # Finally set devices
            if len(final_devices)==1:
                INT_CLASSIFICATION_DEVICE=final_devices[0]
            elif len(final_devices)==2:
                INT_CLASSIFICATION_DEVICE=final_devices[0]
                INT_TOKENIZATION_DEVICE=final_devices[1]
            elif len(final_devices)==3:
                INT_CLASSIFICATION_DEVICE=final_devices[0]
                INT_TOKENIZATION_DEVICE=final_devices[1]
                INT_SEGMENTATION_DEVICE=final_devices[2]

        # On any error in this process default to CPU
        except:
                INT_CLASSIFICATION_DEVICE=-1
                INT_TOKENIZATION_DEVICE=-1
                INT_SEGMENTATION_DEVICE=-1

    if INT_CLASSIFICATION_DEVICE == -1:
        CLASSIFICATION_DEVICE="cpu"
    else:
        CLASSIFICATION_DEVICE="cuda:" + str(INT_CLASSIFICATION_DEVICE)

    if INT_TOKENIZATION_DEVICE == -1:
        TOKENIZATION_DEVICE="cpu"
    else:
        TOKENIZATION_DEVICE="cuda:" + str(INT_TOKENIZATION_DEVICE)

    if INT_SEGMENTATION_DEVICE == -1:
        SEGMENTATION_DEVICE="cpu"
    else:
        SEGMENTATION_DEVICE="cuda:" + str(INT_SEGMENTATION_DEVICE)


    TAG_ENC_TOKENIZER = BertTokenizerFast.from_pretrained("NbAiLab/nb-bert-base")
    SEGMENTATION_TOKENIZER = TAG_ENC_TOKENIZER
    SEGMENTATION_TOKENIZER.model_max_length=512

    MAX_LENGTH_WITHOUT_CLS=SEGMENTATION_TOKENIZER.model_max_length-1

    CLASSIFICATION_MODEL = AutoModelForTokenClassification.from_pretrained(MODELS_DIR + "/classification/")
    CLASSIFICATION_MODEL.to(CLASSIFICATION_DEVICE)
    CLASSIFICATION_MODEL.eval()

    TOKENIZATION_MODEL = AutoModelForTokenClassification.from_pretrained(MODELS_DIR + "/tokenization/")
    TOKENIZATION_MODEL.to(TOKENIZATION_DEVICE)
    TOKENIZATION_MODEL.eval()

    SEGMENTATION_MODEL = AutoModelForTokenClassification.from_pretrained(MODELS_DIR + "/sentence_segmentation/")
    SEGMENTATION_MODEL.to(SEGMENTATION_DEVICE)
    SEGMENTATION_MODEL.eval()

    all_vocab=open(MODELS_DIR + "/sentence_segmentation/vocab.txt","r").read().replace("\r","").split("\n")
    TOKENS_STARTING_WITH_HASH=torch.zeros( len(all_vocab), dtype=torch.bool, device=TOKENIZATION_MODEL.device)

    for i,j in enumerate(all_vocab):
        if j.startswith("##"):
            TOKENS_STARTING_WITH_HASH[i]=True

    torch.no_grad()

    with open(LABEL_ORDER_FILE, "r") as f:
        LABEL_ORDER = json.load(f)

    cmp_key = cmp_to_key(compare_label)

    CLASS_TO_LABEL_NN={}
    with open(LABEL_LIST_FILE, "r") as f:
        label_list= [i for i in f.read().split("\n") if i!=""]
    classes=[]
    with open(LABEL_CLASSES_FILE,"r") as f:
        class_list = [i for i in f.read().split("\n") if i!=""]
    for c in class_list:
        classes=set()
        for i in range(len(c)):
            if c[i]=="1":
                classes.add(label_list[i])
        CLASS_TO_LABEL_NN[c] = classes


    CLASS_TO_LABEL_NN={c:sorted(list(CLASS_TO_LABEL_NN[c] - IGNORE_BERT_TAGS),key=cmp_key) for c in CLASS_TO_LABEL_NN}
    CLASS_TO_LABEL_NN={c:[EQUAL_TAGS[i] if i in EQUAL_TAGS else i for i in CLASS_TO_LABEL_NN[c] ]  for c in CLASS_TO_LABEL_NN }
    CLASS_TO_LABEL_BM={c:[NN_TO_BM[i] if i in NN_TO_BM else i for i in CLASS_TO_LABEL_NN[c]]   for c in CLASS_TO_LABEL_NN}


    MAIN_TAG_LIST_BM=[NN_TO_BM[i] if i in NN_TO_BM else i for i in MAIN_TAG_LIST_NN]

    MAIN_TAG_LIST_DICT_NN={MAIN_TAG_LIST_NN[i]:i for i in range(len(MAIN_TAG_LIST_NN))}
    MAIN_TAG_LIST_DICT_BM={MAIN_TAG_LIST_BM[i]:i for i in range(len(MAIN_TAG_LIST_BM))}

    for i in CLASS_TO_LABEL_NN:
        CLASS_TO_LABEL_NN[i]=[MAIN_TAG_LIST_DICT_NN[j] for j in CLASS_TO_LABEL_NN[i]]

    for i in CLASS_TO_LABEL_BM:
        CLASS_TO_LABEL_BM[i]=[MAIN_TAG_LIST_DICT_BM[j] for j in CLASS_TO_LABEL_BM[i]]

    model_config=json.load(open(MODELS_DIR + "/sentence_segmentation/config.json","r"))
    ID2LABEL=model_config["id2label"]
    ID2LABEL={i:"bm" if ID2LABEL[i]==BOKMAL_LABEL else "nn" if ID2LABEL[i]==NYNORSK_LABEL else "" for i in ID2LABEL}

    with open(NN_FULLFORM_LIST_PATH, "rb") as handle:
        NN_FULLFORM_LIST = pickle.load(handle)

    with open(BM_FULLFORM_LIST_PATH, "rb") as handle:
        BM_FULLFORM_LIST = pickle.load(handle)


# Recursive function to seek the lemma of the word
# 1. Check if the word is 1 character. If yes return None (not found)
# 2. Check if the word class matches. If exclude the first character and try again
# 3. If the word and the type matches and if there is only one option (as tring) return that
# 4. If There are multiple options score them, and pick the most scored one
#            Scoring is done according to the number of matching tags
# 5. Otherwise exclude the first character and try again

# First this function is called to check gen tag. It removes 's ' and s at the end of the word then
# runs the actual lemma function
def get_lemma(word,indice, tags,LIST):
    global GEN_TAG
    global SUBST_TAG
    global PROP_TAG
    global SECOND_PERSON_TAG

    if GEN_TAG in tags:
        if word.endswith("'s") or word.endswith("'S"):
            word=word[:-2]
            if len(tags)==3:
                ss=set(tags)
                if SUBST_TAG in ss and PROP_TAG in ss:
                    return word
            lem=get_lemma_after_check(word,indice,tags,LIST)
            if lem==None:
                return word
        elif word.endswith("s") or word.endswith("S") or word.endswith("'"):
            word=word[:-1]
            if len(tags)==3:
                ss=set(tags)
                if SUBST_TAG in ss and PROP_TAG in ss:
                    return word
            lem=get_lemma_after_check(word,indice,tags,LIST)
            if lem==None:
                return word

    # Check if høflig
    if word=="De":
        if SECOND_PERSON_TAG in tags:
            return "De"
        else:
            return "de"
    return get_lemma_after_check(word,indice,tags,LIST)

def get_lemma_after_check(word, indice, tags, LIST):
    global SUBST_TAG
    global PROP_TAG


#    global general_counter ###
#    if indice==1:
#        print(general_counter)
#        general_counter+=1

    # Set hard limit for recursion. Handy when words are not recognized by the tokenizer
    if indice>300:
        return None

    # If the word is only one character return None
    if len(word[indice:])<=1:
        return None

    # If the word only has subst and prop as tags return the rest of the word as lemma for the rest
    if len(tags)==2 and (tags[0]==SUBST_TAG and tags[1]==PROP_TAG or tags[0]==PROP_TAG and tags[1]==SUBST_TAG):
        return word[indice:]

    pot=LIST.get(str(word[indice:]))
    if pot==None:
        returned=get_lemma_after_check(word, indice+1, tags, LIST)
        if returned==None:
            return None
        return word[indice:indice+1] + returned
    else:
        typ=pot.get(tags[0])
        if typ==None:
            if word[indice:indice+1].isupper():
                word=word[:indice] + word[indice].lower() + word[indice+1:]
                returned = get_lemma_after_check(word, indice, tags, LIST)
                if returned==None:
                    return None
                return returned
            else:
                returned = get_lemma_after_check(word, indice+1, tags, LIST)
                if returned==None:
                    return None
                return word[indice:indice+1] + returned
        else:
            if type(typ)==str:
                return typ
            elif type(typ)==dict:
                scores={i:len(set(typ[i]).intersection(tags[1:])) for i in typ}
                return max(scores, key=scores.get)
            else :
                returned = get_lemma_after_check(word, indice+1, tags, LIST)
                if returned==None:
                    return None
                return word[indice:indice+1] + returned

def get_lemma_for_the_first_word(word, tags, LIST):
    global PROP_TAG
    global SUBST_TAG
    global GEN_TAG
    global SECOND_PERSON_TAG

    if len(word)==1:
        if word=="I" or word=="i":
           return "i"
        if word=="Å" or word=="å":
           return "å"

    # If the word only has subst and prop as tags return the rest of the word as lemma for the rest
    if len(tags)==2 and (tags[0]==SUBST_TAG and tags[1]==PROP_TAG or tags[0]==PROP_TAG and tags[1]==SUBST_TAG):
        return word
    elif len(tags)==3:
        ss=set(tags)
        if SUBST_TAG in ss and PROP_TAG in ss and GEN_TAG in ss:
            if word.endswith("'s") or word.endswith("'S"):
                word=word[:-2]
                return word
            elif word.endswith("s") or word.endswith("S") or word.endswith("'"):
                word=word[:-1]
                return word

    # Check if høflig
    if word=="De":
        if SECOND_PERSON_TAG in tags:
            return "De"
        else:
            return "de"

    pot=LIST.get(word)
    if pot==None:
        if(word[0].isupper()):
            new_word=str(word[0:1].lower()) + str(word[1:])
            return get_lemma(new_word,0,tags,LIST)
        return get_lemma(word,0,tags,LIST)
    else:
        return get_lemma(word,0,tags,LIST)


def matcher(o):
    return o.group(0)[0] + "\n\n" + o.group(0)[2]

def split_titles(txt):
    return [i.replace("\n"," ") for i in re.sub(r"[^.!\?](\n)([^a-z,æ,ø,å,\\ ])", matcher, txt).split("\n\n")]

# This function is called from web iinterfaces instead of tag()
# This function will first do the title analysis and then tag the sentences.
def tag_text(text , write_output_to,  given_lang="au", output_tsv=False, write_identified_lang_to=None, return_as_object=False):
    strs=split_titles(text.strip().replace("\r",""))
    return [tag(s, write_output_to, given_lang, output_tsv, write_identified_lang_to, return_as_object) for s in strs]

# Keeping this function in comment jsut in case tags don't include kvant
#def is_numeric_value(s):
#    if s[0]=="-" and s[1].isnumeric():
#        return s[1:].replace(",","").replace(".","").isnumeric()
#    elif s[0].isnumeric():
#        return s.replace(",","").replace(".","").isnumeric()
#    return False

def tag(text , write_output_to,  given_lang="au", output_tsv=False, write_identified_lang_to=None, return_as_object=False, sentences_splitted=False):
    global SEGMENTATION_TOKENIZER
    global SEGMENTATION_DEVICE
    global SEGMENTATION_MODEL
    global CLASSIFICATION_MODEL
    global ID2LABEL
    global MODEL_SENTENCE_START_ID
    global MODEL_SENTENCE_END_ID
    global MAX_LENGTH_WITHOUT_CLS
    global BATCH_SIZE
    global LANGUAGE_IDENTIFICATIOR_BATCH_SIZE

    global BOKMAL_LABEL
    global NYNORSK_LABEL

    global BOKMAL_LABEL_ID
    global NYNORSK_LABEL_ID

    global TOKENS_STARTING_WITH_HASH

    global PUNCTUATION

    global CLASS_TO_LABEL_NN
    global CLASS_TO_LABEL_BM

    global NN_FULLFORM_LIST
    global BM_FULLFORM_LIST

    global MAIN_TAG_LIST_NN
    global MAIN_TAG_LIST_BM
    global UKJENT_TAG

    global INT_TOKENIZATION_DEVICE


    if sentences_splitted:
        text=[j for j in [i.strip() for i in text] if j!=""]
        encodings = SEGMENTATION_TOKENIZER(text,add_special_tokens=True, padding=False, truncation=True)#.to(SEGMENTATION_MODEL.device)
        sentence_list = encodings["input_ids"]

        # Determine the languge. If the language is given as parameter use that.
        if given_lang=="bm" or given_lang=="au":
            if write_identified_lang_to!=None:
                write_identified_lang_to.write("bm")
            CLASS_TO_LABEL=CLASS_TO_LABEL_BM
            FULLFORM_LIST=BM_FULLFORM_LIST
            MAIN_TAG_LIST=MAIN_TAG_LIST_BM
        else:
            if write_identified_lang_to!=None:
                write_identified_lang_to.write("nn")
            CLASS_TO_LABEL=CLASS_TO_LABEL_NN
            FULLFORM_LIST=NN_FULLFORM_LIST
            MAIN_TAG_LIST=MAIN_TAG_LIST_NN
    else:
#    global general_counter_all

        all_tags_object=[]

        # Just to empty anything allocated on GPU.
        torch.cuda.empty_cache()

        # Here we get the whole text tokenized.
        text=text.replace("\n", " ")
        encodings = SEGMENTATION_TOKENIZER(text,add_special_tokens=False, return_tensors="pt").to(SEGMENTATION_MODEL.device)

        # Save a copy of the tokenization
        original_encodings=copy.deepcopy(encodings)
        original_encodings=original_encodings.to("cpu")
        torch.cuda.empty_cache()

        # Pad to the complete size (model max_size -1 (-1 to add CLS))
        old_size=encodings["input_ids"][0].size()[0]

        # Pad size
        pad_size=MAX_LENGTH_WITHOUT_CLS - old_size%MAX_LENGTH_WITHOUT_CLS

        # Number of rows
        row_count=int(old_size/MAX_LENGTH_WITHOUT_CLS) + 1

        # Do padding with Zeros to the pad_size that we have calculated.
        encodings["input_ids"] = torch.nn.functional.pad(input=encodings["input_ids"], pad=(0, pad_size), mode="constant", value=0)

        # Set the last token as SENTENCE END (SEP)
        encodings["input_ids"][0][old_size]=MODEL_SENTENCE_END_ID

        # Chunk into max_length items
        encodings["input_ids"]=torch.reshape(encodings["input_ids"],(row_count,MAX_LENGTH_WITHOUT_CLS))

        # Add CLS to each item
        encodings["input_ids"]=torch.cat(( torch.full((row_count,1),MODEL_SENTENCE_START_ID, device=SEGMENTATION_MODEL.device) ,encodings["input_ids"]),dim=1)

        # Create attention mask
        encodings["attention_mask"]=torch.ones_like(encodings["input_ids"], device=SEGMENTATION_MODEL.device)

        # Create batches
        input_ids_batched=torch.split(encodings["input_ids"], LANGUAGE_IDENTIFICATIOR_BATCH_SIZE)
        attention_mask_batched=torch.split(encodings["attention_mask"], LANGUAGE_IDENTIFICATIOR_BATCH_SIZE)

        encodings=encodings.to("cpu")
        torch.cuda.empty_cache()

        # Set the last chunk's attention mask according to its size
        attention_mask_batched[-1][-1][pad_size +1:] = 0

        # Now pass all chunks through the model and get the labels
        # While passing, we count the number of bokmal and nynorsk markers
        labels_output=[]
        labels_ids=[0,0,0]

        # First get them back to CPU to open space on GPU
        input_ids_batched=[i.to("cpu") for i in input_ids_batched]
        attention_mask_batched=[i.to("cpu") for i in attention_mask_batched]
        torch.cuda.empty_cache()

        for input_ids, attention_masks in zip(input_ids_batched, attention_mask_batched):
#torch.tensor(b_input_ids).to(device).long()
            current_batch={"input_ids":input_ids.to(SEGMENTATION_MODEL.device).long(), "attention_mask":attention_masks.to(SEGMENTATION_MODEL.device).long()}
            outputs = SEGMENTATION_MODEL(**current_batch)
            del current_batch
            torch.cuda.empty_cache()

            label_data=outputs.logits.argmax(-1)

            label_counts_in_this_chunk=label_data.unique(return_counts=True)
            for l_id, num in zip(label_counts_in_this_chunk[0].tolist(), label_counts_in_this_chunk[1].tolist()):
                if l_id!=0:
                    labels_ids[l_id]+=num
            labels_output.extend(label_data)


        # Determine the languge. If the language is given as parameter use that.
        # If not, use labels_ids to determine the language
        if given_lang=="au" or given_lang==None:
            if labels_ids[1]>labels_ids[2]:
                if write_identified_lang_to!=None:
                    write_identified_lang_to.write("nn")
                CLASS_TO_LABEL=CLASS_TO_LABEL_NN
                FULLFORM_LIST=NN_FULLFORM_LIST
                MAIN_TAG_LIST=MAIN_TAG_LIST_NN
            else:
                if write_identified_lang_to!=None:
                    write_identified_lang_to.write("bm")
                CLASS_TO_LABEL=CLASS_TO_LABEL_BM
                FULLFORM_LIST=BM_FULLFORM_LIST
                MAIN_TAG_LIST=MAIN_TAG_LIST_BM
        elif given_lang=="bm":
            if write_identified_lang_to!=None:
                write_identified_lang_to.write("bm")
            CLASS_TO_LABEL=CLASS_TO_LABEL_BM
            FULLFORM_LIST=BM_FULLFORM_LIST
            MAIN_TAG_LIST=MAIN_TAG_LIST_BM
        else:
            if write_identified_lang_to!=None:
                write_identified_lang_to.write("nn")
            CLASS_TO_LABEL=CLASS_TO_LABEL_NN
            FULLFORM_LIST=NN_FULLFORM_LIST
            MAIN_TAG_LIST=MAIN_TAG_LIST_NN


        # Serialize back
        labels_output=torch.stack(labels_output ,dim=0)
        torch.cuda.empty_cache()
        labels_output=labels_output[:, range(1,MAX_LENGTH_WITHOUT_CLS+1)]
        torch.cuda.empty_cache()
        labels_output=torch.reshape(labels_output,(1,row_count *MAX_LENGTH_WITHOUT_CLS))
        torch.cuda.empty_cache()

        # Now the data is split into sentences
        # So, now create sentence data as list so that this could be used
        # in torch operations and can be input to the models
        sentence_list=[]
        this_sentence=[MODEL_SENTENCE_START_ID]
        last_label=-1
        for token, label in zip(original_encodings["input_ids"][0].tolist(), labels_output[0].tolist()):
            if TOKENS_STARTING_WITH_HASH[token]:
                if last_label!=0:
                    if len(sentence_list)>0:
                        sentence_list[-1].append(token)
                    else:
                        sentence_list=[[token]]
#                        sentence_list.append([token])
                else:
                    this_sentence.append(token)
                    last_label=label
            elif label==0:
                this_sentence.append(token)
                last_label=label
            else:
                this_sentence.append(token)
                sentence_list.append(this_sentence)
                this_sentence=[MODEL_SENTENCE_START_ID]
                last_label=label

        if len(this_sentence)>1:
            sentence_list.append(this_sentence)

        # Remove any tensors from the GPU since we have sentences in the memory now
        del original_encodings
        del labels_output
        del attention_mask_batched
        del input_ids_batched
        del encodings
        del old_size
        del text
        del outputs
        torch.cuda.empty_cache()

    # Create batches
    batched_sentences=[]
    my_batch=[]
    num_sentences=0
    for sentence in sentence_list:
        sentence.append(MODEL_SENTENCE_END_ID)
        if num_sentences==BATCH_SIZE:
            max_len=len(max(my_batch, key=len))
            if max_len>SEGMENTATION_TOKENIZER.model_max_length:
                max_len=SEGMENTATION_TOKENIZER.model_max_length

            my_attentions=torch.LongTensor([[1] * len(i[0:max_len]) + [0]*(max_len-len(i[0:max_len])) for i in my_batch]).to("cpu")
            my_batch=[i[0:max_len] + [0]*(max_len-len(i[0:max_len])) for i in my_batch]
            to_append={
                                    "input_ids": torch.LongTensor(my_batch).to("cpu"),
                                    "attention_mask": my_attentions,
                                    }
            batched_sentences.append(to_append)
            num_sentences =0
            my_batch=[]
        else:
            my_batch.append(sentence)
            num_sentences+=1

    if len(my_batch)>0:
        max_len=len(max(my_batch, key=len))
        if max_len>SEGMENTATION_TOKENIZER.model_max_length:
            max_len=SEGMENTATION_TOKENIZER.model_max_length
        my_attentions=torch.LongTensor([[1] * len(i[0:max_len]) + [0]*(max_len-len(i[0:max_len])) for i in my_batch]).to("cpu")
        my_batch=[i[0:max_len] + [0]*(max_len-len(i[0:max_len])) for i in my_batch]
        to_append={
                            "input_ids": torch.LongTensor(my_batch).to("cpu"),
                            "attention_mask": my_attentions,
                            }
        batched_sentences.append(to_append)

    torch.cuda.empty_cache()

    # Now use the classification model to tag
    # and tokenization model to merge tokens
    for my_batch in batched_sentences:
        my_batch={"input_ids":my_batch["input_ids"].to(CLASSIFICATION_MODEL.device), "attention_mask":my_batch["attention_mask"].to(CLASSIFICATION_MODEL.device)}
        outputs = CLASSIFICATION_MODEL(**my_batch)
        classification_output=outputs.logits.argmax(-1)
        if CLASSIFICATION_MODEL.device!=TOKENIZATION_MODEL.device:
            my_batch["input_ids"]=my_batch["input_ids"].to(TOKENIZATION_MODEL.device)
            my_batch["attention_mask"]=my_batch["attention_mask"].to(TOKENIZATION_MODEL.device)
        outputs = TOKENIZATION_MODEL(**my_batch)
        tokenization_output=outputs.logits.argmax(-1)

        if INT_TOKENIZATION_DEVICE!=-1:
            my_batch["input_ids"]=my_batch["input_ids"].to("cpu")
            my_batch["attention_mask"]=my_batch["attention_mask"].to("cpu")
            classification_output=classification_output.to("cpu")
            tokenization_output=tokenization_output.to("cpu")
            outputs.logits=outputs.logits.to("cpu")
            torch.cuda.empty_cache()

        for i in range(int(classification_output.size()[0])):
            classes = [CLASS_TO_LABEL[ CLASSIFICATION_MODEL.config.id2label[t.item()] ] if CLASSIFICATION_MODEL.config.id2label[t.item()] in CLASS_TO_LABEL else "" for t in classification_output[i]]
            tag=[]
            prepend_to_next=False
            for j,k,l in zip(my_batch["input_ids"][i], tokenization_output[i], classes):
                if j==MODEL_SENTENCE_START_ID:
                    continue
                elif j==MODEL_SENTENCE_END_ID:
                    break
                if TOKENS_STARTING_WITH_HASH[j]:
                    prepend_to_next=False
                    if len(tag)>0:
                        tag[-1]["w"] += SEGMENTATION_TOKENIZER.decode(j)[2:]
                    else:
                        tag=[{"w":SEGMENTATION_TOKENIZER.decode(j)[2:]}]
                elif prepend_to_next:
                    prepend_to_next=False
                    if len(tag)>0:
                        tag[-1]["w"] += SEGMENTATION_TOKENIZER.decode(j)
                    else:
                        tag=[{"w":SEGMENTATION_TOKENIZER.decode(j)}]
                else:
                    tag.append({"w":SEGMENTATION_TOKENIZER.decode(j) , "t":l})
                if k==0:
                    prepend_to_next=True

            # Check if the words come after punctuations. Assign True for their places. False otherwise
            check_for_first_word=[True]+[True if "t" in tagging and len(set(tagging["t"]).intersection(PUNCTUATION))>0 else False for tagging in tag][:-1]
            #or is_numeric_value(tagging["w"])

            # Check if the words that come after punctuations begin with an alphanumeric. True if yes, False otherwise
            # By other words, this marks the words that needs special handling
            check_for_first_word=[ True if item[0] and item[1]["w"].isalpha() else False for item in zip(check_for_first_word, tag)]

            # Get the tags for the words. If it is a marked word, use get_lemma_for_the_first_word else use get_lemma
            tag=[dict(item[1], **dict({"l":get_lemma(item[1]["w"], 0 , item[1]["t"] if "t" in item[1] else [UKJENT_TAG] ,FULLFORM_LIST)if not item[0] else get_lemma_for_the_first_word(item[1]["w"], item[1]["t"] if "t" in item[1] else [UKJENT_TAG] ,FULLFORM_LIST)  }))   for item in zip(check_for_first_word,tag)  ]

            # Assign word as lemma if lemma is None.
            # Assign tag as ukjent if tag is empty set.
            tag=[{"w":j["w"], "l": j["w"] if j["l"]==None else j["l"] , "t":[ MAIN_TAG_LIST[k] for k in j["t"]] if "t" in j else ["ukjent"] } for j in tag]
#            general_counter_all+=len(tag)
#            print(general_counter_all)
            if output_tsv:
                for www in tag:
                    write_output_to.write(www["w"])
                    write_output_to.write("\t")
                    write_output_to.write(www["l"])
                    write_output_to.write("\t")
                    write_output_to.write(" ".join(www["t"]))
                    write_output_to.write("\n")
                write_output_to.write("\n")
            else:
                if return_as_object:
                    all_tags_object.append(tag)
                else:
                    json.dump(tag,write_output_to)
                    write_output_to.write("\n")
    if return_as_object:
        return all_tags_object

def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                    help="file to process. Output to stdout.", metavar="FILE")
    parser.add_argument("-bm", "--bokmal", dest="spraak", action="store_const",
                        const="bm", default="au",
                    help="Tag Bokmål")
    parser.add_argument("-nn", "--nynorsk", dest="spraak", action="store_const",
                        const="nn", default="au",
                    help="Tag Nynorsk")
    parser.add_argument("-au", "--au", dest="spraak", action="store_const",
                        const="au", default="au",
                    help="Identify the langauge (default)")
    parser.add_argument("-i", "--input-dir", dest="input_dir", type=str,
                        help="Directory to process each file in it. Operates "
                        "recursively. An output directory must be provided for "
                        "use with this option. The language is identified "
                        "automatically for each file if no language is set.",
                        metavar="FILE")
    parser.add_argument("-t", "--tsv", dest="output_tsv", action="store_const",
                        const=True, default=False, help="Output in tab-separated format.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                        help="Directory to output tagging. Adds .json to each "
                        "input file name. Overwrites existing output files. "
                        "Tries to create the directory if it does not exist. "
                        "--input-dir must be provided for use with this option.",
                        metavar="FILE")

    parser.add_argument("-b","--batch-size", action="store", default="8",
                        type=str, required=False,
                        help="Batch size for the GPU processing.")

    parser.add_argument("-lb","--language-identificator-batch-size",
                        action="store", default="4",type=str, required=False,
                        help="Batch size for the GPU processing used in language "
                        "identification. This must be less than the normal batch "
                        "size since the whole input space of the model is utilized.")

    parser.add_argument("-s", "--separated-sentences", action="store_const",
                        const="yes", default="no",
                        help="Skip sentence segmentation and language identification. "
                        "Consider each line as a sentence. The default language is --bm (bokmål)."
                        "So, if nynorsk is the language of the sentences, then --nn must be used.")

    args = parser.parse_args()

    if args.batch_size:
        try:
            BATCH_SIZE=int(args.batch_size)
        except:
            pass

    if args.language_identificator_batch_size:
        try:
            LANGUAGE_IDENTIFICATIOR_BATCH_SIZE = int(args.language_identificator_batch_size)
        except:
            pass

    if args.filename:
        if os.path.isfile(args.filename):
            load_models_and_config()
            if args.separated_sentences == "yes":
                strs=open(args.filename,"r").read().strip().replace("\r","").split("\n")
                spraak=args.spraak
                if spraak=="au":
                    spraak="bm"
                tag(strs, sys.stdout, spraak, args.output_tsv, None, False, True)
            else:
                strs=split_titles(open(args.filename,"r").read().strip().replace("\r",""))
                for s in strs:
                    tag(s, sys.stdout, args.spraak, args.output_tsv)
        else:
            print("The file " + args.filename + " could not be found.")
            exit(1)
    elif args.input_dir and args.output_dir:
            output_suf = ".tsv" if args.output_tsv else ".json"

            if not os.path.isdir(args.input_dir):
                print("The input directory " + args.input_dir  + " could not be found.")
                exit(1)

            os.makedirs(args.output_dir, exist_ok=True)

            load_models_and_config()

            for dir_path, _, files in os.walk(args.input_dir):
                for f in files:
                    f_name = os.path.join(dir_path, f)
                    output_f_name = os.path.join(args.output_dir, f) + output_suf

                    print("Input: " + f_name + ", Output: " + output_f_name)

                    with open(f_name, "r") as infile:
                        with open(output_f_name, "w") as outfile:
                            if args.separated_sentences == "yes":
                                strs=infile.read().strip().replace("\r","").split("\n")
                                spraak=args.spraak
                                if spraak=="au":
                                    spraak="bm"
                                tag(strs, outfile, spraak, args.output_tsv, None, False, True)
                            else:
                                strs=split_titles(infile.read().strip().replace("\r",""))
                                for s in strs:
                                    tag(s, outfile, args.spraak, args.output_tsv)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
