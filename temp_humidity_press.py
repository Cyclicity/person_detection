from sense_hat import SenseHat  # Import the SenseHat library

sense = SenseHat()  # Create an instance of the SenseHat class

temp = sense.get_temperature()  # Get the current temperature from the Sense Hat
humidity = sense.get_humidity()  # Get the current humidity from the Sense Hat
pressure = sense.get_pressure()  # Get the current pressure from the Sense Hat


print("Temperature: {:.2f} °C".format(temp))  # Print the temperature with 2 decimal places
print("Humidity: {:.2f} %".format(humidity))  # Print the humidity with 2 decimal places
print("Pressure: {:.2f} mb".format(pressure))  # Print the pressure with 2 decimal places
