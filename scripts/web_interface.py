# This is a simple web interface wrapper for the tagger.
# When run, it runs a web interface at the local and if allowed at the external IP
# 
import sys
sys.path.append('..')
from flask import Flask, request
from tag import *
import io

# Load Models and config requred for tagging
load_models_and_config()


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    text =""
    tagged=""
    err=""
    lang ="au"
    lang_tag=""
    all_tagged=""
    all_err=""
    if request.method == 'POST':
        #"sentence" in request.post:
        text = str(request.form['sentence']).strip()
        lang = str(request.form['lang'])
        if lang=="au":
            lang="au"
        elif lang=="bm":
            lang="bm"
        else:
            lang="nn"

        output_content = io.StringIO()
        lang_used = io.StringIO()

        tag_text(text, output_content, lang, True, lang_used ) 

        all_tagged = output_content.getvalue()
        lang_tag = lang_used.getvalue()
        output_content.close()
        lang_used.close()

        if lang_tag =="bm":
            lang_tag="Bokmål"
        else:
            lang_tag="Nynorsk"
        all_tagged = all_tagged.replace("<","&lt;").replace(">","&gt;")
        all_err =all_err.replace("<","&lt;").replace(">","&gt;")
    
    return "<html><head><title>OBT eksperiment</title></head><body><h2>Setning:</h2></br><form name=tagger method=post action='/tag-experiment/' onSubmit='document.tagger.btn.value=\"Vent!\";document.tagger.btn.disabled=true;return true;'>Auto<input type='radio' name='lang' value='au'" +(" checked" if lang=="au" else "")+ ">&nbsp;&nbsp;&nbsp;Bokmål<input type='radio' name='lang' value='bm'" +(" checked" if lang=="bm" else "")+ ">&nbsp;&nbsp;&nbsp;Nynorsk<input type='radio' name='lang' value='nn'" +(" checked" if lang=="nn" else "")+ "></br><textarea name=sentence style=\"font-family: 'Courier New', monospace;\" rows=25 cols=100>"+text+"</textarea></br><input type=submit value='Tagg!' name=btn></form></br></br>" + ("Språk: " if lang_tag!="" else "") +lang_tag+"</br><textarea name=drs2 style=\"font-family: 'Courier New', monospace;\" rows=50 cols=100>"+all_tagged+"</textarea></br><pre>"+all_err+"</pre></body></html>"


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=9000,debug=True)
