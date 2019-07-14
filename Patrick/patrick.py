import os

import rename_file

def header():
    print('|===================================================|')
    print('|                Welcome, im Patrick                |')
    print('|               ---------------------               |')
    print('|===================================================|')
    os.system('pause')

def body():
    os.system('cls')
    print('|___________________________________________________|')
    print('| Menu :                                            |')
    print('|___________________________________________________|')
    print("| 1. | Edge Detection Using Canny Method            |")
    print("| 2. | Image Smoothing                              |")
    print("| 3. | Convert Image to Grayscale                   |")
    print("| 4. | Convert Image to HSV                         |")
    print("| 5. | Convert Image to CMY                         |")
    print("| 6. | Convert Image to ycbcr                       |")
    print("| 7. | Face and Eyes Detection                      |")
    print("| 8. | Change Image to Edit                         |")
    print("| 9. | Download Image                               |")
    print("| 10.| Exit                                         |")
    print('|----|----------------------------------------------|')

    return int(input('| <> | Choose Me => '))

if __name__ == "__main__":
    choice = 99
    header()
    while choice != 0:
        choice = body()