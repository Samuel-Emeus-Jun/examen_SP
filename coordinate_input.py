def pedir_coordenadas():
    try:
        input_str = input("Introduce las coordenadas (latitud, longitud) separadas por comas: ")
        lat_str, lon_str = input_str.split(",")
        return[lat_str.strip(), lon_str.strip()]
    except ValueError:
        print("Error: Debes introducir exactamente dos valores separados por comas.")
        return None #pedir_coordenadas()

