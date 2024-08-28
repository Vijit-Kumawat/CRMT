import pyttsx3  # pip install pyttsx3
import speech_recognition as sr  # pip install speech_recognition
from playsound import playsound
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import time
import boto3
from botocore.exceptions import ClientError


haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Specify your AWS credentials
aws_access_key_id = 'aws-access-key-id'
aws_secret_access_key = 'aws-secret-access-key'
region_name = 'ap-south-1'

# Initialize the Polly client with your credentials
polly_client = boto3.client('polly',
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key,
                            region_name=region_name)



engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
newVoiceRate = 145
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate',newVoiceRate)

def speak(audio):
    engine.say(audio)
    print(audio)
    engine.runAndWait()

def takeCommand():
    #     It takes microphone input form the user and returns string output
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 0.5
        # audio = r.listen(source)
        audio = r.listen(source,0,2)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-US')
        print(f"User said: {query}")
    except Exception as e:
        print(e)
        return "None"
    return query


def capture_images(subject_name):
    datasets = 'datasets'
    path = os.path.join(datasets, subject_name)
    
    if not os.path.isdir(path):
        os.makedirs(path)
    
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        return
    
    count = 1
    while count <= 30:
        _, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))
            cv2.imwrite(f'{path}/{count}.png', face_resize)
            count += 1
        
        cv2.imshow('Face Capture', frame)
        key = cv2.waitKey(10)
        if key == 27:  # Press 'ESC' to break the loop
            break
    
    webcam.release()
    cv2.destroyAllWindows()
    print(f"Images for {subject_name} captured successfully.")
    
    
def train_model(subject_name):
    datasets = 'datasets'
    path = os.path.join(datasets, subject_name)
    
    if not os.path.isdir(path):
        print(f"No data found for {subject_name}")
        return
    
    model = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label = 0  # Assuming unique label for each subject
    
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            filepath = os.path.join(path, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label)
    
    faces = np.asarray(faces)
    labels = np.asarray(labels)
    
    model.train(faces, labels)
    model.save(f'{subject_name}_face_recognizer_model.yml')
    print(f"Model for {subject_name} trained and saved successfully.")
    
    
def recognize_face():
    haar_file = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    # Function to load models
    def load_models():
        models = {}
        for model_file in os.listdir():
            if model_file.endswith('_face_recognizer_model.yml'):
                subject_name = model_file.split('_')[0]
                model = cv2.face.LBPHFaceRecognizer_create()
                model.read(model_file)
                models[subject_name] = model
        return models

    # Load all models
    models = load_models()

    # Start the webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        return False

    confidence_threshold = 75  # Set confidence threshold to 75%
    while True:
        _, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))

            name = "Unknown"
            best_confidence = float('inf')

            for subject_name, model in models.items():
                label, confidence = model.predict(face_resize)
                if confidence < best_confidence:
                    best_confidence = confidence
                    name = subject_name

            cv2.putText(frame, f'{name} - {best_confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        key = cv2.waitKey(10)
        if key == 27 or best_confidence < confidence_threshold:  # Press 'ESC' or reach confidence threshold to break the loop
            break

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()

    # Return True if any face is recognized and confidence is below threshold, False otherwise
    return any([name != "Unknown" for name in models.keys()]) and best_confidence < confidence_threshold


def list_s3_buckets():
    # Replace these with your actual AWS access key and secret key
    access_key = "access-key"
    secret_key = "secret-key"
    
    # Create a session using your AWS credentials
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    # Create an S3 client
    s3_client = session.client('s3')
    
    try:
        # List buckets
        response = s3_client.list_buckets()
        
        # Get the list of bucket names
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        
        # Print the bucket names
        print("S3 Buckets:")
        for bucket in buckets:
            # print(bucket)
            speak(bucket)
        speak("Buckets are displayed Successfully")
            
    except Exception as e:
        print(e)
        
def send_email():
    try:
        
        region = "ap-south-1"
        access_key = "access-key"
        secret_key = "secret-key"
        sender_email = "your-sender-email@gmail.com"
        speak("Type your subject you want to set")
        subject = input("Type Subject :- ")
        speak("Type body of the mail.")
        body_text = input("Write Body :- ")
        speak("type your s3 Bucket Name!")
        bucket_name = input("Type s3 Bucket Name :- ")  # Replace with your S3 bucket name
        file_name = "./email.txt"  # Name of the file containing email addresses

        # Establish connection to SES and S3
        ses_client = boto3.client('ses', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        s3_client = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

        # Read email addresses from file stored in S3 bucket
        try:
            obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
            email_content = obj['Body'].read().decode('utf-8')
            recipient_emails = [email.strip() for email in email_content.split('\n') if "@" in email]
        except ClientError as e:
            print("An error occurred while accessing the file:", e.response['Error']['Message'])
        except Exception as e:
            speak("An error occurred while reading the file:"+e)
            

        # Send email to each recipient
        for recipient_email in recipient_emails:
            try:
                # Send email
                response = ses_client.send_email(
                    Destination={
                        'ToAddresses': [
                            recipient_email,
                        ],
                    },
                    Message={
                        'Body': {
                            'Text': {
                                'Charset': 'UTF-8',
                                'Data': body_text,
                            },
                        },
                        'Subject': {
                            'Charset': 'UTF-8',
                            'Data': subject,
                        },
                    },
                    Source=sender_email
                )
                speak("Email sent successfully to:", recipient_email)
            except ClientError as e:
                speak("Failed to send email to", recipient_email, ":", e.response['Error']['Message'])
            except Exception as e:
                speak("An error occurred while sending email to", recipient_email,":",e)
    except Exception as e:
        print(e)
        
def text_to_speech(text, output_file):
    try:
        # Synthesize speech
        print("We have two persons choose :- Justin, Joey")
        voice = input("Type the persons name :- ")
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice,  # Choose a Hindi voice (e.g., Aditi)
            LanguageCode='es-US'  # Specify the language code for Hindi (hi-IN)
        )

        # Save speech to a file
        with open(output_file, 'wb') as audio_file:
            audio_file.write(response['AudioStream'].read())

        print(f"Speech saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    





while True:
    
    print("""
          Voice Commands
    ---------------------------
    -   -> First time setup   -
    -   -> Start Cloud        -
    ---------------------------
    """)
    query = takeCommand().lower()
    count=0
    if "first time setup" in query:
        query = input("Type the first setup key: ")
        
        if "9990" in query:
            subject_name = input("Person Name : ")
            capture_images(subject_name)
            train_model(subject_name)
            speak("New User Created Successfully!")
        else:
            # print("Wrong key Entered!")
            speak("Wrong key Entered!")
        
    
    elif "start cloud" in query:
        if count>0:
            break
        while True:  
            if count>0:
                    break       
            try:  
                if count>0:
                    break   
                speak("starting cloud. Getting things ready for Face identity check.")
                count+=1
                if recognize_face():
                    print("Face recognized!")

                    speak("Welcome, Cloud access granted!")
                    speak("System ready to accept commands!")
                    print("""
                          Voice Commands
                    --------------------------------
                    -   -> Create Instance         -
                    -   -> Show Instance           -
                    -   -> Terminate Instance      -
                    -   -> Create Volume           -
                    -   -> Attach Volume           -
                    -   -> Create User             -
                    -   -> Create Bucket           -
                    -   -> Show Bucket             -
                    -   -> Upload file to Bucket   -
                    -   -> Send Email              -
                    -   -> Text to Speech          -
                    -   -> Show Commands           -
                    --------------------------------
                    """)
                    while True:

                        query = takeCommand().lower()
                        

                        
                                
                        if "show instance" in query:
                                try:
                                    def get_running_instances():
                                        
                                        ec2 = boto3.client("ec2", aws_access_key_id='aws-access-key-id', aws_secret_access_key='aws-secret-access-key', region_name='ap-south-1')
                        
                                        # Describe all running instances
                                        response = ec2.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
                        
                                        # Extract instance names and IDs
                                        instances_info = []
                                        for reservation in response['Reservations']:
                                            for instance in reservation['Instances']:
                                                instance_id = instance['InstanceId']
                                                instance_name = ''
                                                for tag in instance['Tags']:
                                                    if tag['Key'] == 'Name':
                                                        instance_name = tag['Value']
                                                        break
                                                instances_info.append({'Name': instance_name, 'InstanceId': instance_id})
                                        return instances_info
                                    
                                    instances_info = get_running_instances()
                                    count = 0
                                    for instance in instances_info:
                                        count+=1
                                        
                                        speak("Instance Name "+instance['Name'])
                                        print()
                                        print("Instance Name:", instance['Name'])
                                        print("Instance ID:", instance['InstanceId'])
                                        print()
                                    if count == 0:
                                        speak("No Active Instance Found")
                                        
                                except Exception as e:
                                    print(e) 
                                    
                        elif "create instance" in query:
                            try:
                                Value = input("Instance Name :- ")
                                ec2=boto3.resource("ec2", aws_access_key_id='aws-access-key-id', aws_secret_access_key='aws-secret-access-key',region_name='ap-south-1')
                                createInstance=ec2.create_instances(
                                    ImageId="ami-013e83f5abcd6baeb",
                                    MinCount=1,
                                    MaxCount=1,
                                    InstanceType="t2.micro",
                                TagSpecifications = [
                                    {
                                        'ResourceType': 'instance',
                                        'Tags': [
                                            {
                                                'Key': 'Name',
                                                'Value': Value
                                            },
                                        ]
                                    },
                                ]
                            )
                                speak("Successfully launched instance with name {} ".format(Value))
                                speak("here is the instance id")
                                print()
                                print("Instance id is :-", createInstance[0].id)

                            except Exception as e:
                                print(e)
                                    
                        elif "terminate instance" in query:
                            try:
                                def terminate_instance_by_name(instance_name):
                                    ec2 = boto3.resource("ec2", aws_access_key_id='aws-access-key-id', aws_secret_access_key='aws-secret-access-key', region_name='ap-south-1')
                    
                                    # Filter instances by name
                                    instances = ec2.instances.filter(Filters=[{'Name': 'tag:Name', 'Values': [instance_name]}])
                    
                                    # Terminate each instance found
                                    for instance in instances:
                                        instance.terminate()
                                        speak("Instance with name '{}' terminated successfully.".format(instance_name))
                    
                                instance_name = input("Instance Name :- ")
                                terminate_instance_by_name(instance_name) #user given
                              
                            except Exception as e:
                                print(e)
                                
                        elif "create volume" in query:
                            try:
                                volume_name = input("volume_name :- ")
                                size=input("size :- ") #user given
                                zone="ap-south-1b"
                                access_key = 'aws-access-key-id'
                                secret_key = 'aws-secret-access-key'
                                region = 'ap-south-1'
                                ec2_client = boto3.client(
                                    'ec2',
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key,
                                    region_name=region
                                    )
                                response = ec2_client.create_volume(
                                    AvailabilityZone=zone,
                                    Size=int(size),
                                        TagSpecifications=[
                                            {
                                                'ResourceType': 'volume',
                                                'Tags': [
                                                    {'Key': 'Name', 'Value': volume_name} #change volume name with user given
                                                ]
                                            }
                                        ]
                                    )
                                volume_id = response['VolumeId']
                            
                                # print("volume EBS Volume of size " + size + " has been launched in zone" + zone)
                                speak("volume EBS Volume of size " + size + " has been launched in zone" + zone)
                                print("Volume ID :- " + volume_id)
                                
                            except Exception as e:
                                print(e)
                                
                                
                        elif "attach volume" in query:
                            try:
                                instance_name = input("Type Instance Name :-") #user given
                                volume_name = input("Type Volume Name :- ") #user given

                                access_key = 'aws-access-key-id'
                                secret_key = 'aws-secret-access-key'

                                session = boto3.Session(
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key
                                )

                                ec2_client = session.client('ec2', region_name='ap-south-1')

                                # Get the instance ID based on the instance name
                                response = ec2_client.describe_instances(
                                    Filters=[
                                        {'Name': 'tag:Name', 'Values': [instance_name]}
                                    ]
                                )

                                instances = response['Reservations']

                                if not instances:
                                    print("No instances found with the provided name.")
                                    exit()

                                # Assuming only one instance matches the provided name, get its ID
                                instance_id = instances[0]['Instances'][0]['InstanceId']

                                # Get the volume ID based on the volume name
                                response = ec2_client.describe_volumes(
                                    Filters=[
                                        {'Name': 'tag:Name', 'Values': [volume_name]}
                                    ]
                                )

                                volumes = response['Volumes']

                                if not volumes:
                                    speak("No volumes found with the provided name.")
                                    exit()


                                # Assuming only one volume matches the provided name, get its ID
                                volume_id = volumes[0]['VolumeId']

                                # Attach volume to the instance
                                response = ec2_client.attach_volume(
                                    Device='/dev/xvdf',
                                    InstanceId=instance_id,
                                    VolumeId=volume_id
                                )

                                speak(response)
                            except Exception as e:
                                print(e)
                                
                        elif "create user" in query:                         
                            try:
                                name = input("User Name :- ")
                                access_key = 'aws-access-key-id'
                                secret_key = 'aws-secret-access-key'
                                
                                session = boto3.Session(
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key,
                                    region_name='ap-south-1'
                                )
                                
                                iam_client = session.client('iam')
                                
                                username = name  # Replace with your desired IAM username
                                
                                # Step 1: Create IAM User
                                response = iam_client.create_user(
                                    UserName=username
                                )
                                
                                speak(f"IAM user '{username}' created successfully!")
                                
                                # Step 2: Attach Administrative Policy
                                admin_policy_arn = 'arn:aws:iam::aws:policy/AdministratorAccess'  # This is the ARN of the AdministratorAccess policy
                                
                                response_attach = iam_client.attach_user_policy(
                                    UserName=username,
                                    PolicyArn=admin_policy_arn
                                )
                                
                                speak(f"Administrative policy attached successfully to user '{username}'!")
                                
                                # Step 3: Generate Access Key
                                response_key = iam_client.create_access_key(
                                    UserName=username
                                )
                                
                                access_key_id = response_key['AccessKey']['AccessKeyId']
                                secret_access_key = response_key['AccessKey']['SecretAccessKey']
                                
                                speak("Here is the Access Key and the Secret Access Key!")
                                print("Access Key ID:", access_key_id)
                                print("Secret Access Key:", secret_access_key)
                                
                            except Exception as e:
                                print(e)
                                 
                        elif "create bucket" in query:    
                            try:
                                 
                                speak("Type the name of the bucket it must be unique")
                                bucket_name = input("Name of Bucket (Must be Unique!) :").lower()
                                region = "ap-south-1"
                                access_key = "access-key"
                                secret_key = "secret-key"

                                s3_client = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

                                # Create S3 bucket
                                response = s3_client.create_bucket(
                                    Bucket=bucket_name,
                                    CreateBucketConfiguration={
                                        'LocationConstraint': region
                                    }
                                )
                                speak(f"S3 bucket '{bucket_name}' created successfully in region '{region}'.")
                            
                            except Exception as e:
                                speak("Already exist")
                        
                        elif "show buckets" in query:
                            try:
                                list_s3_buckets()
                                                              
                            except Exception as e:
                                print(e)
                                
                        elif "upload file to bucket" in query:
                            try:
                                
                                bucket_name = input("Bucket Name :- ")  # Replace "your_bucket_name_here" with your S3 bucket name
                                region = "ap-south-1"
                                access_key = "access-key"
                                secret_key = "secret-key"
                                file_path = "./upload/"
                                
                                # Establish connection to S3
                                s3_client = boto3.client('s3', region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
                                
                                # Get a list of all files in the "upload" folder
                                files_to_upload = os.listdir(file_path)
                                
                                # Upload each file to S3 bucket
                                for file in files_to_upload:
                                    file_path_with_name = os.path.join(file_path, file)
                                    try:
                                        with open(file_path_with_name, "rb") as file_obj:
                                            s3_client.upload_fileobj(file_obj, bucket_name, file)
                                        speak(f"File {file} uploaded successfully to S3 bucket: {bucket_name}")
                                    except Exception as e:
                                        print(f"Error uploading file {file}: {e}")
                            
                            except Exception as e:
                                print(e)
                        
                        elif "send email" in query:
                            try:
                                send_email()
                                speak("email sent successfully!")
                                
                            except Exception as e:
                                print(e)
                                
                                
                        elif "show commands" in query:
                            print("""
                                  Voice Commands
                            --------------------------------
                            -   -> Create Instance         -
                            -   -> Show Instance           -
                            -   -> Terminate Instance      -
                            -   -> Create Volume           -
                            -   -> Attach Volume           -
                            -   -> Create User             -
                            -   -> Create Bucket           -
                            -   -> Show Bucket             -
                            -   -> Upload file to Bucket   -
                            -   -> Send Email              -
                            -   -> Text to Speech          -
                            -   -> Show Commands           -
                            --------------------------------
                            """)
                                
                        elif "text to speech" in query:
                            try:
                                print("Starting the program...")
                                # Prompt the user to enter the text to convert
                                text = input("Enter the text to convert to speech (in Hinglish or Hindi): ")
                                # Check if text is provided
                                if not text:
                                    print("No text provided.")
                                    
                                # Specify the output file path for the speech
                                output_file = 'output.mp3'
                                # Convert text to speech in Hindi and save to file
                                text_to_speech(text, output_file)
                                
                            except Exception as e:
                                print(e)
                                
                                     
                else:
                    print("Person is unknown. Exiting loop.")
                    break
                
            except Exception as e:
                print("Face Not Recognized")
                speak("Face Not Recognized, Try Again!")
                cv2.VideoCapture(0).release()
                cv2.destroyAllWindows()                
                break
            
            
            

            
            
            
                                
            
            
            
            

            
            

        
            
                
                            
                        
                            
            
                            
            
            
            
            
                        
            
            
            
            
              