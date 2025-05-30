import datetime
import sys

import m.availability
import m.monitor

availabilities=[]
previousAvailabilities=[]

while True:
    try:
        if availabilities:
            previousAvailabilities = availabilities
        availabilities = m.availability.build_availability_dict(sys.argv[1:])
        strChanged = m.monitor.avail_added_removed_Str(previousAvailabilities, availabilities)
        if strChanged:
            current_time = datetime.datetime.now()
            print(datetime.datetime.now(), " :")
            print(strChanged)
    except KeyboardInterrupt:
        break