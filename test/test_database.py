import pymysql

db = pymysql.connect("localhost", "root", "1", "mess")
cursor = db.cursor()


## create table
cursor.execute("DROP TABLE IF EXISTS distribution_networks")

sql_start = """CREATE TABLE distribution_networks ("""
sql = 'SCENARIO  INT NOT NULL,\n TIME INT primary key,\n '
for i in range(32):
    sql += "Pij{0} FLOAT,\n ".format(i)
for i in range(32):
    sql += "Qij{0} FLOAT,\n ".format(i)
for i in range(32):
    sql += "Iij{0} FLOAT,\n ".format(i)
for i in range(33):
    sql += "V{0} FLOAT,\n ".format(i)
for i in range(6):
    sql += "Pg{0} FLOAT,\n ".format(i)
for i in range(5):
    sql += "Qg{0} FLOAT,\n ".format(i)
sql += "Qg{0} FLOAT\n ".format(5)
sql_start_end = """)"""

cursor.execute(sql_start + sql + sql_start_end)

sql_start = "INSERT INTO distribution_networks("
sql="SCENARIO,TIME,"
for i in range(32):
    sql += "Pij{0},".format(i)
for i in range(32):
    sql += "Qij{0},".format(i)
for i in range(32):
    sql += "Iij{0},".format(i)
for i in range(33):
    sql += "V{0},".format(i)
for i in range(6):
    sql += "Pg{0},".format(i)
for i in range(5):
    sql += "Qg{0},".format(i)
sql += "Qg{0}".format(5)
sql+= ") VALUES (" + "0,".__mul__(1+32+32+32+33+6+6)+"0)"

cursor.execute(sql_start + sql)
db.commit()
db.close()

