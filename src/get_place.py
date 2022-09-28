from audioop import add
import pandas as pd

def get_address_data():
    input_book = pd.read_excel('data/garbage_place.xlsx')
    # read address
    address = input_book['住所']
    latitude = input_book['X']
    longitude = input_book['Y']
    return address, latitude, longitude
    
def make_address_list():
    """
        address = [latitude, longitude]
    """
    address_data = get_address_data()
    address = address_data[0]
    latitude = address_data[1]
    longitude = address_data[2]
    
    # if "-" change to "_"
    # elif address contains zenkaku space change to "_"
    if address.str.contains('-').any():
        address = address.str.replace('-', '_')
    if address.str.contains('－').any():
        address = address.str.replace('－', '_')
    if address.str.contains(' ').any():
        address = address.str.replace(' ', '_')
    if address.str.contains('　').any():
        address = address.str.replace('　', '_')
    if address.str.contains('、').any():
        address = address.str.replace('、', '_')
    if address.str.contains('（').any():
        address = address.str.replace('（', '_')
    if address.str.contains('）').any():
        address = address.str.replace('）', '_')
    if address.str.contains('\(').any():
        address = address.str.replace('\(', '_', regex=True)
    if address.str.contains('\)').any():
        address = address.str.replace('\)', '_', regex=True)
    if address.str.contains('・').any():
        address = address.str.replace('・', '_')
    if address.str.contains('･').any():
        address = address.str.replace('･', '_')
    if address.str.contains('‐').any():
        address = address.str.replace('‐', '_')
    if address.str.contains('.').any():
        address = address.str.replace('.', '_', regex=True)
    if address.str.contains('～').any():
        address = address.str.replace('～', 'To')
    if address.str.contains(',').any():
        address = address.str.replace(',', 'To', regex=True)
    if address.str.contains('＝').any():
        address = address.str.replace('＝', '_')
    if address.str.contains('/').any():
        address = address.str.replace('/', '_', regex=True)
        
    
    for i in range(len(address)):
        print("{} = [{}, {}]".format(address[i], latitude[i], longitude[i]))
        
if __name__ == '__main__':
    make_address_list()