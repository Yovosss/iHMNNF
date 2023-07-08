
# coding: utf-8
import os
import sys
import json

from django.http import HttpResponse
from src import infoExtract_forShengFuBao

import os
os.environ["NLS_LANG"] = "GERMAN_GERMANY.UTF8" 
import warnings
warnings.filterwarnings('ignore')

import time

def get_extract_result(request): 

    if request.method == 'GET':
        query_dict = request.GET.copy()
    elif request.method == 'POST':
        query_dict = json.loads(request.POST['data'], encoding = 'utf-8')[0]

    note_type = str(query_dict['note_type'])
    note_text = str(query_dict['note_text'])
    print(note_text)
    procedure_concept_name = None
    if 'procedure_concept_name' in query_dict.keys():
        procedure_concept_name = str(query_dict['procedure_concept_name'])
    with_rest = None
    if 'with_rest' in query_dict.keys() and query_dict['with_rest'].lower() != "false":
        with_rest = str(query_dict['with_rest'])

    ie = infoExtract_forShengFuBao.ie_forShengFuBao()

    try:
        result_dict = ie.info_extract(note_type, note_text, procedure_concept_name, with_rest)
        js = {
            "code": 0,
            "msg": 'success',
            "data": result_dict
        }
    except:
        js = {
            "data": "",
            "msg": "params query incorrect!",
            "code": 1
        }
    return HttpResponse(json.dumps(js,ensure_ascii=False))
