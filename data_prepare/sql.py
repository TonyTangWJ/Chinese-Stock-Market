import mysql.connector
import numpy as np
import pandas as pd
            


class sql:
    def __init__(self):
        self.conn = None
        self.cur = None

    def sql_login(self, database='csm_pro'):
        self.conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Password123',
            database=database,
            use_pure=False,
            allow_local_infile=True
        )
        self.cur = self.conn.cursor()

    def insert_data(self, df, table, database='csm_pro', batch_size=1000):
        self.sql_login(database)
        placeholders = ','.join(['%s'] * len(df.columns))
        sql = f"INSERT INTO {table} VALUES ({placeholders})"
        data = [tuple(row) for row in df.values]
        self.conn.autocommit = False

        try:
            if len(data) <= batch_size:
                self.cur.executemany(sql, data)
                self.conn.commit()
                # print('Data inserted successfully')
            else:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    self.cur.executemany(sql, batch)
                    self.conn.commit()
                # print('Data inserted successfully')
        except Exception as e:
            self.conn.rollback()
            print(f"Data inserted failed: {e}\t")
        finally:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()


    def return_data(self, sql, database='csm_pro'):
        self.sql_login(database)
        try:
            self.cur.execute(sql)
            result = np.array(self.cur.fetchall())
            columns = [desc[0] for desc in self.cur.description]
            return pd.DataFrame(result, columns=columns).replace(r'\s+', '', regex=True)
        finally:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()