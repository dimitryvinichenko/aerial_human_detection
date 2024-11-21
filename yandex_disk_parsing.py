import os
import requests
from urllib.parse import urlencode
import urllib


def download_jpeg(public_link, save_directory, patth, ext, str_num):
    patth = 'Датасет Аэрозрение v.1 — отдельные снимки/02_second_part_DataSet_Human_Rescue/images/'
    final_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download' \
                + '?public_key=' + urllib.parse.quote(public_link) + '&path=/' + urllib.parse.quote(patth+str_num+ext)

    response = requests.get(final_url)
    if response.status_code != 200:
        # print("Ошибка при получении ссылки на файл:", response.json())
        return

    download_url = response.json().get('href')

    # Скачиваем файл
    file_name = os.path.join(save_directory, str_num + ext)  # Укажите нужное расширение
    file_response = requests.get(download_url)
    if file_response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(file_response.content)


def download_file(public_link, save_directory, patth, ext, str_num):
    # Получаем прямую ссылку на файл
    final_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download' \
          + '?public_key=' + urllib.parse.quote(public_link) + '&path=/' + urllib.parse.quote(patth+ext)

    response = requests.get(final_url)
    if response.status_code != 200:
        #print("Ошибка при получении ссылки на файл:", response.json())
        return

    download_url = response.json().get('href')

    # Скачиваем файл
    file_name = os.path.join(save_directory, str_num + ext)  # Укажите нужное расширение
    file_response = requests.get(download_url)

    if file_response.status_code == 200:
        if file_response.content != b'\r\n':
            with open(file_name, 'wb') as f:
                f.write(file_response.content)
            download_jpeg(public_link, save_directory, patth, '.jpg', str_num)
            #print(f"Файл {file_name} успешно скачан.")
    else:
        pass
        #print("Ошибка при скачивании файла:", file_response.json())


if __name__ == "__main__":
    # public_link = 'https://disk.yandex.ru/d/vX2zgJl8-XB0Og'
    public_link = 'https://disk.yandex.ru/d/bN06_6ncEXTLIw'
    patth = 'Новая папка/Датасет Аэрозрение v.1 — отдельные снимки/02_second_part_DataSet_Human_Rescue/labels/'
    save_directory = '/home/dima/data3/'  # Путь к папке куда сохранять

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for number in range(1000):  # 10000 - сколько файлов 
        str_num = str(number).zfill(6)
        temp_path = patth + str_num
        download_file(public_link, save_directory, temp_path, '.txt', str_num)

        print(number)
