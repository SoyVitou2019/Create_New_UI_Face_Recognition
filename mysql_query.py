import mysql.connector
import datetime

class MysqlQuery:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

    def write_data_into_users(self, data):
        cursor = self.conn.cursor()
        insert_query = "INSERT INTO users (username, email, password, shift_id) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_query, data)
        self.conn.commit()
        print("successfully")

    def check_in_or_out(self, user_id):
        current_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
        cursor = self.conn.cursor()
        period = ['00:00:00', '24:00:00']
        query = "SELECT check_status FROM attendances WHERE user_id = %s AND DATE(created_at) = %s AND time(created_at) BETWEEN %s AND %s ORDER BY created_at DESC LIMIT 1"
        cursor.execute(query, (user_id, current_date, period[0], period[1]))
        check_status = cursor.fetchone()
        if check_status:
            return check_status[0]
        else:
            return None
        
    def get_user_id_by_username(self, username):
        cursor = self.conn.cursor()
        query = "SELECT id, shift_id FROM users WHERE CONCAT(LOWER(first_name), '-', LOWER(last_name)) = %s"
        cursor.execute(query, (username,))  
        user_data = cursor.fetchone()
        if user_data:
            user_id, shift_id = user_data
            return user_id, shift_id
        else:
            return None, None
        
    def write_data_into_attendance(self, name):
        user_id, shift_id = self.get_user_id_by_username(username=name)
        # shift_id = int(shift_id)
        check_status = self.check_in_or_out(user_id=user_id)
        if check_status == None:
            check_status = 0
        else:
            check_status = not check_status

        cursor = self.conn.cursor()
        data = [user_id, check_status]
        insert_query = "INSERT INTO attendances (user_id, check_status) VALUES (%s, %s)"
        cursor.execute(insert_query, data)
        self.conn.commit()
        print("successfully")

    def close_connection(self):
        if self.conn.is_connected():
            self.conn.close()
            print("Connection closed.")
        else:
            print("No active connection to close.")
