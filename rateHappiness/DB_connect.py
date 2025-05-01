from pymongo import MongoClient
import source
myclient = MongoClient("mongodb+srv://Pixel:Pixel7788@cluster0.3dpfxx3.mongodb.net/mydb?retryWrites=true&w=majority")
db_name = myclient["DB4200"]

Collection = db_name["out_data"]

document = {
    'Happiness levels': source.outputs
}

# Insert the document into MongoDB
Collection.insert_one(document)

myclient.close()





