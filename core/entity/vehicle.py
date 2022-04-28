
import datetime
from core.common import common







class Criminal:
    def __init__(self, id, type_criminal,velocity,type_vehicle, image, license_plate):
        self.type_criminal = type_criminal
        self.type_vehicle = type_vehicle
        self.image = common.encodemIMG(image)
        self.license_plate = license_plate
        self.datetime = datetime.datetime.now()
        self.id = self.datetime.strftime('%m %d %Y %H %M %S %p').replace(' ', '')+str(id)
        self.velocity = velocity







class Criminal_entity:
    def __init__(self, id, type_criminal,velocity,type_vehicle, blob_img, license_plate, datetime):
        self.type_criminal = type_criminal
        self.type_vehicle = type_vehicle
        self.image = common.decodemIMG(blob_img)
        self.license_plate = license_plate
        self.datetime = datetime
        self.id = id
        self.velocity = velocity
    

class Criminal_fGUI:
    def __init__(self, id, type_criminal,velocity,type_vehicle, img, license_plate, datetime):
        self.type_criminal = type_criminal
        self.type_vehicle = type_vehicle
        self.image = img
        self.license_plate = license_plate
        self.datetime = datetime
        self.id = id
        self.velocity = velocity

