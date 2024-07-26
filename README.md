# analytics code for citizen assemblies
This is an analytics project for any services that need deep learning, machine learning and data science. 

The endpoints can be consumed and used by anybody for the purposes of creating other services like websites and mobile apps 

The goal for all these is educating and sharing information from and to citizens of Kenya on matters that involve their country.

check the website at [citizenassemblies.com](https://citizenassemblies.com)

## how to run the project
- get your huggingface api key and add it to the.env file
- run the project with docker compose using the command `docker-compose up --build`

## how to contribute to the project
- fork the project
- clone the project
- create a new branch
- make your changes
- create a pull request

## current endpoints to use to create websites and mobile apps for sharing information.

## examples of how to use the endpoints
 - curl
    
   ```
    curl -X POST "http://64.71.146.81:8010/ask" -H "Content-Type: application/json" -d '{"question": "What is democracy?", "max_tokens": 256}'
    
    ```

- python

        
   ```
    import requests
    import json

    url = "http://64.71.146.81:8010/ask"
    headers = {
        "Content-Type": "application/json"
    }

    max_tokens = 256 
    question = "How can education help people?"  
    payload = {
        "question": question,
        "max_tokens": 256,
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    print("Status Code:", response.status_code)
    print("Response Body:", response.text)

   ```


