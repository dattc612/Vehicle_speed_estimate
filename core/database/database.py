
import sqlite3
import cv2
from core.common import common
import cv2

class Database:
    DB_LOCATION = 'core/database/criminal.db'
    def __init__(self):
        print('================================START INITIALIZATION DATABASE===================================')
        self.conn = sqlite3.connect(Database.DB_LOCATION,  check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("""CREATE TABLE if not exists tblVehicle(
            id text INTEGER NOT NULL,
            type_vehicle text,
            velocity real, 
            type_criminal text NOT NULL,
            image BLOB ,
            license_plate text,
            datetime timestam,
            PRIMARY KEY(id, type_criminal)
        )""")
        self.conn.commit()
        print('================================END INITIALIZATION DATABASE===================================')

    def close(self):
        self.conn.close()

    def insert(self, criminal):
        sqlite_insert_with_param  = 'INSERT  INTO tblVehicle VALUES (?,?,?,?,?,?,?)'
        data_tuple = (criminal.id,criminal.type_vehicle, criminal.velocity,criminal.type_criminal,criminal.image,criminal.license_plate,criminal.datetime)
        self.cur.execute(sqlite_insert_with_param,data_tuple)
        # self.cur.execute('insert into {} (t1, i1) values (?, ?)'.format(self._table), (row['t1'], row['i1']))
        self.conn.commit()
    def update(self,criminal):
        sqlite_insert_with_param = 'update tblVehicle set type_vehicle = ?, velocity = ?, type_criminal = ?, image = ?, license_plate = ?, datetime = ? where id = ?'
        data_tuple =  (criminal.type_vehicle, criminal.velocity,criminal.type_criminal,criminal.image,criminal.license_plate,criminal.datetime,criminal.id)
        self.cur.execute(sqlite_insert_with_param, data_tuple)
        self.conn.commit()
    def update_license_plate(self,id, type_criminal,license_plate):
        sqlite_insert_with_param = 'UPDATE tblVehicle SET  license_plate = ? WHERE id = ? AND type_criminal = ?'
        data_tuple =  (license_plate, id, type_criminal)
        self.cur.execute(sqlite_insert_with_param, data_tuple)
        self.conn.commit() 
    def get_all(self):
        sqlite_insert_with_param  = 'select * from tblVehicle'
        self.cur.execute(sqlite_insert_with_param)
        results = self.cur.fetchall()
        self.conn.commit()
        for i in range(len(results)):
            results[i]= list(results[i])
            results[i][4] =  cv2.cvtColor(common.decodemIMG(results[i][4]), cv2.COLOR_BGR2RGB)
            if results[i][5] == '':
                results[i][5] = common.Full_Detect(results[i][4])
                try:
                    self.update_license_plate(results[i][0],results[i][3], results[i][5])
                except Exception as e:
                    print(e)

        return results





