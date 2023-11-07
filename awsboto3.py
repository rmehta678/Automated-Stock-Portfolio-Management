import boto3
import json
import uuid

def make_boto_sesh():
    session = boto3.Session(aws_access_key_id='AKIARMK4EY3E2KLJ4RUT', 
                            aws_secret_access_key='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',
                            region_name='us-east-1')

    # Verify that the session is valid
    if session.get_credentials():
        print("Session is valid")
    else:
        print("Session is not valid")

    return session

def send_sqs_message(session):
    # Get the SQS client
    sqs = session.client('sqs')

    #Dedup ID
    message_deduplication_id = uuid.uuid4().hex

    # Create a message body
    message_body = json.dumps({'stock_data': {'AAPL': 255}, 'uid': message_deduplication_id})

    # Message group
    message_group_id = 'finnhub'

    # Send the message to the queue
    queue_url = 'https://sqs.us-east-1.amazonaws.com/095219402441/finhubb.fifo'
    sqs.send_message(QueueUrl=queue_url, MessageBody=message_body, MessageGroupId=message_group_id, MessageDeduplicationId=message_deduplication_id)

# def store_sqs_message_in_s3(mess)
def main():
    session = make_boto_sesh()
    send_sqs_message(session)

if __name__ == '__main__':
    main()