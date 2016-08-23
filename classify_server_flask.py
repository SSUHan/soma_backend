from flask import Flask, request, jsonify, abort
from sklearn.externals import joblib
from preprocess import clean_title, pp_kkma, pp_twitter
import logging


# Load models
clf = joblib.load('classify.model')
cate_dict = joblib.load('cate_dict.dat')
vectorizer = joblib.load('vectorizer.dat')

# cate_id_name_dict = dict(map(lambda k,v:(v,k),cate_dict.items()))
cate_id_name_dict = dict((k, v) for v, k in cate_dict.items())


# Initialize app
app = Flask(__name__)


print("Server Started.")

@app.route('/')
def index():
    return "this is soma backend flask server page"

@app.route('/classify')
def classify():
        print("#")
        img_url = request.args.get('img', '')
        name = request.args.get('name', '')
        print('> Name: %s' % name)

        pred = clf.predict(vectorizer.transform([name]))[0]
        return jsonify({'cate': cate_id_name_dict[pred]})


if __name__ == '__main__':
        app.run(host='0.0.0.0', port=18888)