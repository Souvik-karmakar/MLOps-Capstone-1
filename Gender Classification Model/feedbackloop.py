import logging

# Initialize logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def log_feedback(feedback_data):
    try:
        with open('feedback.log','a') as feedback_file:
            feedback_file.write(f'{feedback_data}\n')
        logger.info("User feedback logged successfully")    
    except Exception as e:
        logger.error(f"Error logging user feedback: {str(e)}")    