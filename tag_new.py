import os
import argparse
import json
import re
import sys
import torch
import copy
import pickle
from functools import cmp_to_key
from transformers import AutoModelForTokenClassification, AutoTokenizer

BATCH_SIZE=8
LANGUAGE_IDENTIFICATIOR_BATCH_SIZE=1
MAX_LENGTH=512
TOKENIZER="ltg/norbert3-large"
SEGMENTATION_DEVICE="cuda:0"
CLASSIFICATION_DEVICE="cuda:0"
SEGMENTATION_MODEL=None
CLASSIFICATION_MODEL=None
SCRIPT_PATH=os.path.abspath(os.path.dirname(__file__))
MODELS_DIR=SCRIPT_PATH + "/models"
LABEL_LIST_FILE = MODELS_DIR + "/label_list.txt"
LABEL_CLASSES_FILE = MODELS_DIR + "/labels_classifier.txt"
LABEL_ORDER_FILE = MODELS_DIR + "/labels_order.json"
SEGMENTATION_MODEL_DIR = MODELS_DIR + "/sentence_segmentation/"
CLASSIFICATION_MODEL_DIR = MODELS_DIR + "/classification/"
NN_FULLFORM_LIST_PATH = SCRIPT_PATH + "/nn.pickle"
BM_FULLFORM_LIST_PATH = SCRIPT_PATH + "/bm.pickle"
LABEL_ORDER = None
CLASS_TO_LABEL_BM = None
CLASS_TO_LABEL_NN = None
MAIN_TAG_LIST_BM = None
EQUAL_TAGS = None
ID2LABEL = None
BOKMAL_LABEL="B"
NYNORSK_LABEL="N"
BOKMAL_LABEL_ID=1
NYNORSK_LABEL_ID=2
TOKEN_MERGES = None
PUNCTUATION = None
PUNCTUATION_TAG_LIST = set(["<anf>", "<komma>", "<parentes-beg>", "<parentes-slutt>", "<strek>", "<kolon>", "<punkt>", "<spm>", "<utrop>", "kvant", "symb"])
NN_TO_BM ={
        "høfleg":"høflig",
        "eint":"ent",
        "<ikkje-clb>": "<ikke-clb>",
        "<ordenstal>": "<ordenstall>",
        "<romartal>" : "<romertall>",
        "bu": "be",
        "<st-verb>": "<s-verb>"
        }

MAX_LENGTH_WITHOUT_CLS = MAX_LENGTH - 1
MODEL_SENTENCE_START_ID = None
MODEL_SENTENCE_END_ID = None
NN_FULLFORM_LIST=None
BM_FULLFORM_LIST=None

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
    global TOKENIZER
    global SEGMENTATION_DEVICE
    global CLASSIFICATION_DEVICE
    global SEGMENTATION_MODEL
    global CLASSIFICATION_MODEL
    global MAX_LENGTH
    global CLASS_TO_LABEL_BM
    global CLASS_TO_LABEL_NN
    global MAIN_TAG_LIST_BM
    global MAIN_TAG_LIST_NN
    global EQUAL_TAGS
    global NN_TO_BM
    global ID2LABEL
    global BOKMAL_LABEL
    global NYNORSK_LABEL
    global LABEL_ORDER
    global LABEL_CLASSES_FILE
    global LABEL_LIST_FILE
    global SEGMENTATION_MODEL_DIR
    global CLASSIFICATION_MODEL_DIR
    global MODEL_SENTENCE_START_ID
    global MODEL_SENTENCE_END_ID
    global NN_FULLFORM_LIST
    global BM_FULLFORM_LIST
    global NN_FULLFORM_LIST_PATH
    global BM_FULLFORM_LIST_PATH
    global TOKEN_MERGES
    global PUNCTUATION
    global PUNCTUATION_TAG_LIST
    
    with open(LABEL_ORDER_FILE, "r") as f:
        LABEL_ORDER = json.load(f)
    CLASS_TO_LABEL_NN={}
    with open(LABEL_LIST_FILE, "r") as f:
        MAIN_TAG_LIST_NN = [i for i in f.read().split("\n") if i!=""]
    classes=[]
    with open(LABEL_CLASSES_FILE,"r") as f:
        class_list = [i for i in f.read().split("\n") if i!=""]

    for index, c in enumerate(class_list):
        classes=set()
        for i in range(len(c)):
            if c[i]=="1":
                classes.add(MAIN_TAG_LIST_NN[i])
        CLASS_TO_LABEL_NN[index] = classes

    cmp_key = cmp_to_key(compare_label)

    CLASS_TO_LABEL_NN={c:sorted(list(CLASS_TO_LABEL_NN[c]),key=cmp_key) for c in CLASS_TO_LABEL_NN}
    CLASS_TO_LABEL_BM={c:[NN_TO_BM[i] if i in NN_TO_BM else i for i in CLASS_TO_LABEL_NN[c]]   for c in CLASS_TO_LABEL_NN}


    MAIN_TAG_LIST_BM=[NN_TO_BM[i] if i in NN_TO_BM else i for i in MAIN_TAG_LIST_NN]

    MAIN_TAG_LIST_DICT_NN={MAIN_TAG_LIST_NN[i]:i for i in range(len(MAIN_TAG_LIST_NN))}
    MAIN_TAG_LIST_DICT_BM={MAIN_TAG_LIST_BM[i]:i for i in range(len(MAIN_TAG_LIST_BM))}

    for i in CLASS_TO_LABEL_NN:
        CLASS_TO_LABEL_NN[i]=[MAIN_TAG_LIST_DICT_NN[j] for j in CLASS_TO_LABEL_NN[i]]

    for i in CLASS_TO_LABEL_BM:
        CLASS_TO_LABEL_BM[i]=[MAIN_TAG_LIST_DICT_BM[j] for j in CLASS_TO_LABEL_BM[i]]

    SEGMENTATION_MODEL = AutoModelForTokenClassification.from_pretrained(SEGMENTATION_MODEL_DIR, trust_remote_code=True)
    SEGMENTATION_MODEL.to(SEGMENTATION_DEVICE)
    SEGMENTATION_MODEL.eval()

    CLASSIFICATION_MODEL = AutoModelForTokenClassification.from_pretrained(CLASSIFICATION_MODEL_DIR, trust_remote_code=True)
    CLASSIFICATION_MODEL.to(CLASSIFICATION_DEVICE)
    CLASSIFICATION_MODEL.eval()

    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    MODEL_SENTENCE_START_ID = int(TOKENIZER.convert_tokens_to_ids("[CLS]"))
    MODEL_SENTENCE_END_ID  = int(TOKENIZER.convert_tokens_to_ids("[SEP]"))

    with open(NN_FULLFORM_LIST_PATH, "rb") as handle:
        NN_FULLFORM_LIST = pickle.load(handle)

    with open(BM_FULLFORM_LIST_PATH, "rb") as handle:
        BM_FULLFORM_LIST = pickle.load(handle)
    
    TOKEN_MERGES = [TOKENIZER.decode(i).startswith(" ")  for i in range(len(TOKENIZER))]
    for i,j in enumerate(MAIN_TAG_LIST_NN):
        print(i,j)
    exit(0)
    print(MAIN_TAG_LIST_NN)
    exit(0)
#    PUNCTUATION 
#PUNCTUATION_TAG_LIST

#    model_config=json.load(open(MODELS_DIR + "/sentence_segmentation/config.json","r"))
#    ID2LABEL=model_config["id2label"]
#    ID2LABEL={i:"bm" if ID2LABEL[i]==BOKMAL_LABEL else "nn" if ID2LABEL[i]==NYNORSK_LABEL else "" for i in ID2LABEL}



def tag(text , write_output_to,  given_lang="au", output_tsv=False, write_identified_lang_to=None, return_as_object=False, sentences_splitted=False):
    global TOKENIZER
    global SEGMENTATION_DEVICE
    global SEGMENTATION_MODEL
    global CLASSIFICATION_MODEL
    global CLASSIFICATION_DEVICE
    global MAX_LENGTH_WITHOUT_CLS
    global MODEL_SENTENCE_START_ID
    global MODEL_SENTENCE_END_ID
    global NN_FULLFORM_LIST
    global BM_FULLFORM_LIST
    global MAX_LENGTH
    global LANGUAGE_IDENTIFICATIOR_BATCH_SIZE
    global BATCH_SIZE
    global TOKEN_MERGES

    # Just to empty anything allocated on GPU.
    torch.cuda.empty_cache()

    if sentences_splitted:
        text=[j for j in [i.strip() for i in text] if j!=""]
        encodings = TOKENIZER(text,add_special_tokens=True, padding=False, truncation=True)#.to(SEGMENTATION_MODEL.device)
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

        # Here we get the whole text tokenized.
        text=text.replace("\n", " ")
        encodings = TOKENIZER(text,add_special_tokens=False, return_tensors="pt").to(SEGMENTATION_MODEL.device)

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
        #last_label=-1
        for token, label in zip(original_encodings["input_ids"][0].tolist(), labels_output[0].tolist()):
            #if TOKENS_STARTING_WITH_HASH[token]:
            #    if last_label!=0:
            #        if len(sentence_list)>0:
            #            sentence_list[-1].append(token)
            #        else:
            #            sentence_list=[[token]]
#                        sentence_list.append([token])
            #    else:
            #        this_sentence.append(token)
            #        last_label=label
            if label==0:
                this_sentence.append(token)
                #last_label=label
            else:
                this_sentence.append(token)
                sentence_list.append(this_sentence)
                this_sentence=[MODEL_SENTENCE_START_ID]
                #last_label=label

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
            if max_len>MAX_LENGTH:
                max_len=MAX_LENGTH

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
        if max_len>MAX_LENGTH:
            max_len=MAX_LENGTH
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

        # Done using model on the model device. Bring tensors back to CPU
        classification_output=classification_output.to("cpu")
        my_batch["input_ids"]=my_batch["input_ids"].to("cpu")
        my_batch["attention_mask"]=my_batch["attention_mask"].to("cpu")

        # Go over results for all sentences and get words and assign their classes
        for i in range(int(classification_output.size()[0])):
            classes = [CLASS_TO_LABEL[int(t)]  for t in classification_output[i]]
            tag = []
            for token, token_class in zip(my_batch["input_ids"][i], classes):
                if token==MODEL_SENTENCE_START_ID:
                    continue
                elif token==MODEL_SENTENCE_END_ID:
                    break
                if TOKEN_MERGES[token]:
                    tag.append({"w":TOKENIZER.decode(token)[1:], "t":token_class})
                else:
                    tag[-1]["w"] += TOKENIZER.decode(token)


            # Check if the words come after punctuations. Assign True for their places. False otherwise
            check_for_first_word=[True]+[True if "t" in tagging and len(set(tagging["t"]).intersection(PUNCTUATION))>0 else False for tagging in tag][:-1]
            #or is_numeric_value(tagging["w"])

            # Check if the words that come after punctuations begin with an alphanumeric. True if yes, False otherwise
            # By other words, this marks the words that needs special handling
            check_for_first_word=[ True if item[0] and item[1]["w"].isalpha() else False for item in zip(check_for_first_word, tag)]

            # Get the tags for the words. If it is a marked word, use get_lemma_for_the_first_word else use get_lemma
            tag=[dict(item[1], **dict({"l":get_lemma(item[1]["w"], 0 , item[1]["t"] if "t" in item[1] else [UKJENT_TAG] ,FULLFORM_LIST)if not item[0] else get_lemma_for_the_first_word(item[1]["w"], item[1]["t"] if "t" in item[1] else [UKJENT_TAG] ,FULLFORM_LIST)  }))   for item in zip(check_for_first_word,tag)  ]

            print(tag)

    exit(0)

def matcher(o):
    return o.group(0)[0] + "\n\n" + o.group(0)[2]

def split_titles(txt):
    return [i.replace("\n"," ") for i in re.sub(r"[^.!\?](\n)([^a-z,æ,ø,å,\\ ])", matcher, txt).split("\n\n")]

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
