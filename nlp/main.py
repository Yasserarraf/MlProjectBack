import json

from flask_cors import CORS, cross_origin;
from flask import Blueprint, jsonify

from .extensions import mongo
from nlp.services.scraping.scraping import *
from nlp.services.sentimentAnalysis.model import *
from bson import json_util

main = Blueprint('main', __name__)
CORS(main, resources={r"/saveData": {"origins": "*"}})


@main.route('/')
def index():
    # user_collection = mongo.db.users
    # user_collection.insert({'name': 'yasser'})
    # data = getFirstRecord()
    # del data['_id']
    # result = pd.DataFrame(data,index=[0])
    # print(result)

    return '<a href="/sentimentAnalysis">sentimentAnalysis</a>'


@main.route('/scraping', methods=['GET', 'OPTIONS'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def saveData():
    links = getLinks()
    mainScraping(links)

    article_collection = mongo.db.articles
    # article_collection.delete_many({})
    for article in data:
        article_collection.insert(article)
    print(data)


@main.route('/data', methods=['GET', 'OPTIONS'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def getRecords():
    data_articles = mongo.db.articles.find()

    return json.dumps(list(data_articles), default=json_util.default)


@main.route('/sentimentAnalysis', methods=['GET', 'OPTIONS'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def sentimentAnalysis():

    data = mongo.db.articles.find()
    print("/n******************************/n")
    result = pd.DataFrame(data)
    del result['_id']
    final_result = sentiment_analysis_process(result)
    f = json.loads(final_result)
    print(f['feature_extraction'])
    return final_result