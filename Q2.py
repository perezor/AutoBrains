import pandas as pd
import matplotlib.pyplot as plt


def plot(x, y, title):
    plt.plot(x, y)
    plt.xlabel("time_sec (seconds)")
    plt.ylabel("distance (meters)")
    plt.title(title)
    plt.savefig(title)
    plt.show()


if __name__ == '__main__':
    detection_csv = "Q2_detection_file_2.csv"

    # 1. Load the detection file 'Q2_detction_file_2.csv'
    df = pd.read_csv(detection_csv)

    # 2. Plot the distance of the detected object as a function of time
    plt.plot(df['time_sec'], df['rw_kinematic_point_z'])
    plot(df['time_sec'], df['rw_kinematic_point_z'], title="Distance Over Time")

    # 3. Detect and remove the invalid distance measurements
    df_clean = df[df['rw_kinematic_point_z'] > 0]
    plot(df_clean['time_sec'], df_clean['rw_kinematic_point_z'], title="Distance Over Time (clean)")

    # 4. Calculate the velocity of the object
    last_row = df_clean.iloc[-1]
    last_distance = last_row['rw_kinematic_point_z']
    last_time = last_row['time_sec']
    approximate_velocity = last_distance * last_time
    print("Approximate constant velocity: {}m/sec".format(approximate_velocity))

    # 5. Bonus Question
    detection_csv_bonus = "Q2_detection_file_2_bonus.csv"
    df_bonus = pd.read_csv(detection_csv_bonus)

    # remove entries where both before and after values change drastically (10% up or down)
    df_bonus['prev'] = df_bonus['rw_kinematic_point_z'].shift(1)
    df_bonus['next'] = df_bonus['rw_kinematic_point_z'].shift(-1)
    df_bonus_clean = df_bonus[~(
            ((df_bonus['rw_kinematic_point_z'] > df_bonus['prev']*1.1) & (df_bonus['rw_kinematic_point_z'] > df_bonus['next']*1.1)) |
            ((df_bonus['rw_kinematic_point_z'] < df_bonus['prev']*0.9) & (df_bonus['rw_kinematic_point_z'] < df_bonus['next']*0.9)))]
    plot(df_bonus_clean['time_sec'], df_bonus_clean['rw_kinematic_point_z'], title="Distance Over Time (bonus)")
