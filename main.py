from fastapi import FastAPI, Depends, HTTPException, status, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler, BeforeValidator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing import List, Optional, Dict, Any, Union, Annotated
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from bson import ObjectId
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from fastapi.encoders import jsonable_encoder
import uuid
import requests
from enum import Enum
import httpx
from pymongo.collection import Collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

required_env_vars = [
    "MONGO_URI",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
    "SECRET_KEY",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "GMAIL_ADDRESS",
    "AIRTABLE_API_KEY",
    "AIRTABLE_BASE_ID",
    "AIRTABLE_TABLE_NAME"
]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["naija_concierge"]

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")


FLUTTERWAVE_SECRET_KEY = os.getenv("FLUTTERWAVE_SECRET_KEY")
FLUTTERWAVE_BASE_URL = "https://api.flutterwave.com/v3"
FLUTTERWAVE_SECRET_HASH = os.getenv("FLUTTERWAVE_SECRET_HASH")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
EXCHANGE_RATE_BASE_URL = "https://v6.exchangerate-api.com/v6"


# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

app = FastAPI(title="Naija Concierge API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

smtp_server = "smtp.gmail.com"
smtp_port = 465
email_address = "ahuekweprinceugo@gmail.com"  # Make sure this matches your sending address
password = os.getenv("GMAIL_PASS")  # Use the environment variable
recipient_email = "ahuekweprinceugo@gmail.com" #change to a test email

print(password)
subject = "Test Email from Python"  # Added a subject
message_text = "This is a test email sent using smtplib and a context manager."  # Added message

msg = MIMEText(message_text)
msg['Subject'] = subject
msg['From'] = email_address
msg['To'] = recipient_email

try:
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.ehlo()  # Identify ourselves to the server
        # server.starttls()  # Put the connection in TLS mode
        # server.ehlo()  # Re-identify after TLS negotiation
        server.login(email_address, password)
        server.sendmail(email_address, [recipient_email], msg.as_string())
    print("Email sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: GetJsonSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.str_schema()

# PyObjectId = Annotated[str, BeforeValidator(str)]

class UserBase(BaseModel):
    email: EmailStr
    firstName: str
    lastName: str
    phone: Optional[str] = None
    address: Optional[str] = None
    profileImage: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    profileImage: Optional[str] = None


class UserInDB(UserBase):
    # id: Optional[Annotated[PyObjectId, Field(alias="_id", default=None)]]
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    role: str = "user"
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    hashed_password: str

    model_config = {
        "populate_by_name": True,
        "json_encoders": {
            ObjectId: str
        }
    }

    # Model validators for Pydantic v2
    @classmethod
    def model_validate(cls, obj, **kwargs):
        # Convert ObjectId to string if present
        if isinstance(obj, dict) and '_id' in obj and isinstance(obj['_id'], ObjectId):
            obj['_id'] = str(obj['_id'])
        return super().model_validate(obj, **kwargs)


class User(UserBase):
    id: str
    role: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: User


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None


class ServiceBase(BaseModel):
    name: str
    description: str
    category: str
    price: float
    image: Optional[str] = None
    duration: str
    isAvailable: bool = True


class ServiceCreate(ServiceBase):
    pass


class ServiceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    image: Optional[str] = None
    duration: Optional[str] = None
    isAvailable: Optional[bool] = None


class ServiceInDB(ServiceBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Service(ServiceBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


class CRMClientBase(BaseModel):
    clientName: str
    contactInfo: Dict[str, str]
    serviceBooked: str
    status: str = "pending"
    assignedVendor: Optional[str] = None
    notes: Optional[str] = None
    dueDate: Optional[datetime] = None

class CRMClientCreate(CRMClientBase):
    pass

class CRMClientUpdate(BaseModel):
    clientName: Optional[str] = None
    contactInfo: Optional[Dict[str, str]] = None
    serviceBooked: Optional[str] = None
    status: Optional[str] = None
    assignedVendor: Optional[str] = None
    notes: Optional[str] = None
    dueDate: Optional[datetime] = None


class CRMClientInDB(CRMClientBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class CRMClient(CRMClientBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True

class AirtableBookingForm(BaseModel):
    clientName: str
    email: EmailStr
    phone: Optional[str] = None
    serviceId: str
    bookingDate: datetime
    specialRequests: Optional[str] = None


# class Package(BaseModel):
#     id: str
#     name: str
#     description: str
#     price: float  # Price in NGN
#     duration: int
#     features: List[str]
#     image: Optional[str]
#     type: str
#     isPopular: bool
#     createdAt: datetime
#     updatedAt: datetime

class PackageBase(BaseModel):
    name: str
    description: str
    price: float
    duration: str
    features: List[str]
    image: Optional[str] = None
    type: str
    isPopular: bool = False


class PackageCreate(PackageBase):
    pass


class PackageUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    duration: Optional[str] = None
    features: Optional[List[str]] = None
    image: Optional[str] = None
    type: Optional[str] = None
    isPopular: Optional[bool] = None


class PackageInDB(PackageBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Package(PackageBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class BookingBase(BaseModel):
    userId: str
    serviceId: str
    bookingDate: datetime
    status: str = "pending"
    specialRequests: Optional[str] = None


class BookingCreate(BookingBase):
    pass


class BookingUpdate(BaseModel):
    bookingDate: Optional[datetime] = None
    status: Optional[str] = None
    specialRequests: Optional[str] = None


class BookingInDB(BookingBase):
    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Booking(BookingBase):
    id: str
    createdAt: datetime
    updatedAt: datetime
    service: Optional[Service] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class SubscriptionStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    cancelled = "cancelled"


class SubscriptionInitiate(BaseModel):
    userId: str
    packageId: str
    preferredCurrency: Optional[str] = "NGN"




class SubscriptionBase(BaseModel):
    userId: str
    packageId: str
    startDate: datetime
    endDate: datetime
    status: str = "active"


class SubscriptionCreate(SubscriptionBase):
    pass


class SubscriptionUpdate(BaseModel):
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    status: Optional[SubscriptionStatus]


class SubscriptionInDB(SubscriptionBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Subscription(SubscriptionBase):
    id: str
    createdAt: datetime
    updatedAt: datetime
    package: Optional[Package] = None

    class Config:
        orm_mode = True


class BlogBase(BaseModel):
    title: str
    slug: str
    content: str
    excerpt: str
    coverImage: Optional[str] = None
    author: Dict[str, str]
    tags: List[str]


class BlogCreate(BlogBase):
    pass


class BlogUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    content: Optional[str] = None
    excerpt: Optional[str] = None
    coverImage: Optional[str] = None
    author: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None


class BlogInDB(BlogBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Blog(BlogBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class EmergencyAlertBase(BaseModel):
    userId: str
    message: str
    location: Optional[str] = None
    status: str = "pending"


class EmergencyAlertCreate(EmergencyAlertBase):
    pass


class EmergencyAlertUpdate(BaseModel):
    message: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None


class EmergencyAlertInDB(EmergencyAlertBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class EmergencyAlert(EmergencyAlertBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    subject: str
    message: str


class DocumentBase(BaseModel):
    userId: str
    name: str
    type: str
    url: str


class DocumentCreate(DocumentBase):
    pass


class DocumentInDB(DocumentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    uploadDate: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Document(DocumentBase):
    id: str
    uploadDate: datetime

    class Config:
        orm_mode = True


# class TransactionCreate(BaseModel):
#     subscriptionId: str
#     amount: float
#     currency: str
#     transactionId: str
#     status: str

class TransactionCreate(BaseModel):
    tx_ref: str
    transactionId: str
    userId: str
    packageId: str
    amount: float  # Amount in NGN
    currency: str  # Base currency (NGN)
    preferredCurrency: str  # User's preferred currency
    status: str


class TransactionInDB(TransactionCreate):
    createdAt: datetime
    updatedAt: datetime

class Transaction(TransactionInDB):
    id: str



class BookingDataPoint(BaseModel):
    name: str
    bookings: int
    completed: int

class RevenueDataPoint(BaseModel):
    name: str
    revenue: float

class ChartDataResponse(BaseModel):
    bookingData: List[BookingDataPoint]
    revenueData: List[RevenueDataPoint]


class Timeframe(str, Enum):
    weekly = "weekly"
    monthly = "monthly"


# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(email: str):
    user = db.users.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])  # Convert ObjectId to string
        return UserInDB(**user)
    return None


# def get_user_by_id(user_id: str):
#     try:
#         user = db.users.find_one({"_id": ObjectId(user_id)})
#         if user:
#             return UserInDB(**user)
#     except Exception as e:
#         logger.error(f"Error getting user by ID: {e}")
#     return None

def get_user_by_id(user_id: str):
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            logger.error(f"User not found for ID: {user_id}")
            return None
        user["_id"] = str(user["_id"])  # Convert ObjectId to string
        return UserInDB(**user)
    except ValidationError as ve:
        logger.error(f"Validation error for user ID {user_id}: {ve}")
        return None
    except Exception as e:
        logger.error(f"Error getting user by ID {user_id}: {e}", exc_info=True)
        return None


def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    return current_user


async def get_admin_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


def send_email(to_email: str, subject: str, html_content: str):
    """Send email using SMTP, with improved connection handling."""
    smtp_server = os.getenv("SMTP_HOST")  # Use SMTP_HOST
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("GMAIL_ADDRESS")  # Use GMAIL_ADDRESS, ensure it's the sender

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email

    html_part = MIMEText(html_content, "html")
    message.attach(html_part)

    server = None  # Initialize server outside the try block
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        # server.starttls()  # Upgrade to secure connection
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, message.as_string())
        logger.info(f"Email sent successfully to {to_email}")  # Add logging
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False
    finally:
        if server:
            try:
                server.quit()  # Ensure server.quit() is always called
            except Exception as e:
                logger.error(f"Error closing SMTP connection: {e}")


def send_admin_notification(subject: str, message: str):
    """
    Send an email notification to all admin users.

    Args:
        subject (str): Email subject
        message (str): Email body (plain text or HTML)
        db (Collection): MongoDB users collection

    Raises:
        HTTPException: If no admins are found or email sending fails
    """
    try:
        # Find all admin users
        admin_users = db.users.find({"role": "admin"})
        admin_emails = [user["email"] for user in admin_users]

        if not admin_emails:
            logger.warning("No admin users found for notification")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No admin users found to notify"
            )

        # Send email to each admin
        for email in admin_emails:
            try:
                send_email(email, subject, message)
                logger.info(f"Notification sent to admin: {email}")
            except Exception as e:
                logger.error(f"Failed to send notification to {email}: {e}")
                continue  # Continue with next admin if one fails

        logger.info(f"Notifications sent to {len(admin_emails)} admins")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending admin notifications: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send admin notifications"
        )

def upload_file_to_cloudinary(file: UploadFile, folder: str = "naija_concierge"):
    """Upload file to Cloudinary and return the URL"""
    try:
        result = cloudinary.uploader.upload(
            file.file,
            folder=folder,
            public_id=f"{uuid.uuid4()}",
            overwrite=True
        )
        return result.get("secure_url")
    except Exception as e:
        logger.error(f"Error uploading to Cloudinary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )


def serialize_object_id(obj):
    """Convert ObjectId to string in a dictionary"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                obj[k] = str(v)
            elif isinstance(v, dict):
                serialize_object_id(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        serialize_object_id(item)
    return obj


def add_to_airtable(booking_data: Dict) -> Dict:
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('AIRTABLE_API_KEY')}",
            "Content-Type": "application/json"
        }
        url = f"https://api.airtable.com/v0/{os.getenv('AIRTABLE_BASE_ID')}/{os.getenv('AIRTABLE_TABLE_NAME')}"

        # Ensure all values are JSON-serializable
        serialized_data = {
            "records": [{
                "fields": {
                    key: str(value) if isinstance(value, (ObjectId, datetime)) else value
                    for key, value in booking_data.items()
                }
            }]
        }
        logger.info(f"Sending to Airtable: {serialized_data}")

        response = requests.post(url, headers=headers, json=serialized_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json() if e.response.content else str(e)
        logger.error(f"Airtable HTTP error: {e}, Response: {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Airtable API error: {error_detail}"
        )
    except Exception as e:
        logger.error(f"Error adding to Airtable: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add to Airtable: {str(e)}"
        )

async def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{EXCHANGE_RATE_BASE_URL}/{EXCHANGE_RATE_API_KEY}/latest/{from_currency}"
            )
            response.raise_for_status()
            rates = response.json().get("conversion_rates", {})
            return rates.get(to_currency, 1.0)
        except Exception as e:
            logger.error(f"Error fetching exchange rate: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch exchange rate")


async def initialize_flutterwave_payment(
        email: str,
        amount: float,
        currency: str,
        tx_ref: str,
        country: str
):
    url = "https://api.flutterwave.com/v3/payments"
    headers = {
        "Authorization": f"Bearer {os.getenv('FLUTTERWAVE_SECRET_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "tx_ref": tx_ref,
        "amount": amount,
        "currency": currency,
        "redirect_url": os.getenv("FLUTTERWAVE_REDIRECT_URL", "https://yourapp.com/redirect"),
        "customer": {
            "email": email
        },
        "customizations": {
            "title": "Subscription Payment",
            "description": "Payment for subscription package"
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for non-200 responses
        return response.json()

async def verify_flutterwave_payment(transaction_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        try:
            headers = {"Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}"}
            response = await client.get(
                f"{FLUTTERWAVE_BASE_URL}/transactions/{transaction_id}/verify",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error verifying Flutterwave payment: {e}")
            raise HTTPException(status_code=500, detail="Failed to verify payment")

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    computed_hash = hmac.new(
        FLUTTERWAVE_SECRET_HASH.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed_hash, signature)

async def create_subscription_from_transaction(transaction: dict):
    try:
        package = db.packages.find_one({"_id": ObjectId(transaction["packageId"])})
        if not package:
            logger.error(f"Package not found: {transaction['packageId']}")
            return None

        active_subscription = db.subscriptions.find_one({
            "userId": transaction["userId"],
            "status": "active"
        })
        if active_subscription:
            logger.error(f"User {transaction['userId']} already has an active subscription")
            return None

        subscription_in_db = SubscriptionInDB(
            userId=transaction["userId"],
            packageId=transaction["packageId"],
            startDate=datetime.utcnow(),
            endDate=datetime.utcnow() + timedelta(days=package["duration"]),
            status=SubscriptionStatus.active,
            createdAt=datetime.utcnow(),
            updatedAt=datetime.utcnow()
        )
        result = db.subscriptions.insert_one(subscription_in_db.dict(by_alias=True))
        created_subscription = db.subscriptions.find_one({"_id": result.inserted_id})

        package_obj = Package(
            id=str(package["_id"]),
            name=package["name"],
            description=package["description"],
            price=package["price"],
            duration=package["duration"],
            features=package["features"],
            image=package.get("image"),
            type=package["type"],
            isPopular=package["isPopular"],
            createdAt=package["createdAt"],
            updatedAt=package["updatedAt"]
        )

        user = get_user_by_id(transaction["userId"])
        if user:
            subscription_html = f"""
            <html>
                <body>
                    <h1>Subscription Confirmation</h1>
                    <p>Dear {user["firstName"]},</p>
                    <p>Your subscription to the {package["name"]} package has been successfully created.</p>
                    <p>Subscription Details:</p>
                    <ul>
                        <li>Package: {package["name"]}</li>
                        <li>Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}</li>
                        <li>End Date: {created_subscription["endDate"].strftime("%Y-%m-%d")}</li>
                        <li>Price: {transaction["preferredCurrency"]} (equivalent to NGN {transaction["amount"]})</li>
                        <li>Status: {created_subscription["status"]}</li>
                    </ul>
                    <p>Thank you for choosing Naija Concierge. If you have any questions, please contact us.</p>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            send_email(user["email"], "Subscription Confirmation - Naija Concierge", subscription_html)

        notification_message = f"""
        New subscription created:
        - Client: {user["firstName"]} {user["lastName"]}
        - Package: {package["name"]}
        - Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}
        - Status: {created_subscription["status"]}
        - Amount Paid: {transaction["preferredCurrency"]} (equivalent to NGN {transaction["amount"]})
        """
        send_admin_notification("New Subscription Created", notification_message)

        return Subscription(
            id=str(created_subscription["_id"]),
            userId=created_subscription["userId"],
            packageId=created_subscription["packageId"],
            startDate=created_subscription["startDate"],
            endDate=created_subscription["endDate"],
            status=created_subscription["status"],
            createdAt=created_subscription["createdAt"],
            updatedAt=created_subscription["updatedAt"],
            package=package_obj
        )
    except Exception as e:
        logger.error(f"Error creating subscription from transaction: {e}")
        return None




# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Naija Concierge API"}


# Auth routes
@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    # Check if user already exists
    if db.users.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    del user_dict["password"]
    user_in_db = UserInDB(
        **user_dict,
        hashed_password=hashed_password,
        role="user"
    )

    # Insert into database
    result = db.users.insert_one(user_in_db.dict(by_alias=True))

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    # Get created user
    created_user = db.users.find_one({"_id": result.inserted_id})
    user_response = User(
        id=str(created_user["_id"]),
        email=created_user["email"],
        firstName=created_user["firstName"],
        lastName=created_user["lastName"],
        phone=created_user.get("phone"),
        address=created_user.get("address"),
        profileImage=created_user.get("profileImage"),
        role=created_user["role"],
        createdAt=created_user["createdAt"],
        updatedAt=created_user["updatedAt"]
    )

    # Send welcome email
    welcome_html = f"""
    <html>
        <body>
            <h1>Welcome to Naija Concierge, {user.firstName}!</h1>
            <p>Thank you for registering with us. We're excited to help you experience Lagos like never before.</p>
            <p>If you have any questions or need assistance, please don't hesitate to contact us.</p>
            <p>Best regards,<br>The Naija Concierge Team</p>
        </body>
    </html>
    """
    send_email(user.email, "Welcome to Naija Concierge", welcome_html)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    user_response = User(
        id=str(user.id),
        email=user.email,
        firstName=user.firstName,
        lastName=user.lastName,
        phone=user.phone,
        address=user.address,
        profileImage=user.profileImage,
        role=user.role,
        createdAt=user.createdAt,
        updatedAt=user.updatedAt
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.get("/auth/me", response_model=User)
async def get_me(current_user: UserInDB = Depends(get_current_active_user)):
    return User(
        id=str(current_user.id),
        email=current_user.email,
        firstName=current_user.firstName,
        lastName=current_user.lastName,
        phone=current_user.phone,
        address=current_user.address,
        profileImage=current_user.profileImage,
        role=current_user.role,
        createdAt=current_user.createdAt,
        updatedAt=current_user.updatedAt
    )


# User routes
@app.get("/users", response_model=List[User])
async def get_users(
        skip: int = 0,
        limit: int = 100,
        current_user: UserInDB = Depends(get_admin_user)
):
    users = list(db.users.find().skip(skip).limit(limit))
    return [
        User(
            id=str(user["_id"]),
            email=user["email"],
            firstName=user["firstName"],
            lastName=user["lastName"],
            phone=user.get("phone"),
            address=user.get("address"),
            profileImage=user.get("profileImage"),
            role=user["role"],
            createdAt=user["createdAt"],
            updatedAt=user["updatedAt"]
        ) for user in users
    ]


@app.get("/users/{user_id}", response_model=User)
async def get_user_details(
        user_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Only admins can view other users' details
    if str(current_user.id) != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return User(
        id=str(user.id),
        email=user.email,
        firstName=user.firstName,
        lastName=user.lastName,
        phone=user.phone,
        address=user.address,
        profileImage=user.profileImage,
        role=user.role,
        createdAt=user.createdAt,
        updatedAt=user.updatedAt
    )


@app.put("/users/{user_id}", response_model=User)
async def update_user(
        user_id: str,
        user_update: UserUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Only the user themselves or an admin can update a user
    if str(current_user.id) != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Check if user exists
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update user
    update_data = user_update.dict(exclude_unset=True)
    if update_data:
        update_data["updatedAt"] = datetime.utcnow()
        result = db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User update failed",
            )

    # Get updated user
    updated_user = get_user_by_id(user_id)

    return User(
        id=str(updated_user.id),
        email=updated_user.email,
        firstName=updated_user.firstName,
        lastName=updated_user.lastName,
        phone=updated_user.phone,
        address=updated_user.address,
        profileImage=updated_user.profileImage,
        role=updated_user.role,
        createdAt=updated_user.createdAt,
        updatedAt=updated_user.updatedAt
    )


@app.delete("/users/{user_id}")
async def delete_user(
        user_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        bookings = db.bookings.find_one({"userId": user_id})
        if bookings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete user with existing bookings",
            )

        subscriptions = db.subscriptions.find_one({"userId": user_id})
        if subscriptions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete user with existing subscriptions",
            )

        result = db.users.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User deletion failed",
            )

        return {"message": "User deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User deletion failed",
        )


@app.post("/users/profile-image")
async def upload_profile_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_current_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/profiles")

    result = db.users.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"profileImage": image_url, "updatedAt": datetime.utcnow()}}
    )

    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update profile image",
        )

    return {"profileImage": image_url}


# crm

@app.get("/crm/clients", response_model=List[CRMClient])
async def get_crm_clients(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_admin_user)
):
    query = {}
    if status:
        query["status"] = status

    clients = list(db.crm_clients.find(query).skip(skip).limit(limit))
    return [
        CRMClient(
            id=str(client["_id"]),
            clientName=client["clientName"],
            contactInfo=client["contactInfo"],
            serviceBooked=client["serviceBooked"],
            status=client["status"],
            assignedVendor=client.get("assignedVendor"),
            notes=client.get("notes"),
            dueDate=client.get("dueDate"),
            createdAt=client["createdAt"],
            updatedAt=client["updatedAt"]
        ) for client in clients
    ]


@app.post("/crm/clients", response_model=CRMClient)
async def create_crm_client(
        client: CRMClientCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    if not client.clientName.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client name is required"
        )
    if not client.contactInfo.get("email") and not client.contactInfo.get("phone"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one contact method (email or phone) is required"
        )

    try:
        service = db.services.find_one({"_id": ObjectId(client.serviceBooked)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )
    except Exception as e:
        logger.error(f"Invalid service ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid service ID"
        )

    client_in_db = CRMClientInDB(**client.dict())
    result = db.crm_clients.insert_one(client_in_db.dict(by_alias=True))

    created_client = db.crm_clients.find_one({"_id": result.inserted_id})

    return CRMClient(
        id=str(created_client["_id"]),
        clientName=created_client["clientName"],
        contactInfo=created_client["contactInfo"],
        serviceBooked=created_client["serviceBooked"],
        status=created_client["status"],
        assignedVendor=created_client.get("assignedVendor"),
        notes=created_client.get("notes"),
        dueDate=created_client.get("dueDate"),
        createdAt=created_client["createdAt"],
        updatedAt=created_client["updatedAt"]
    )


@app.get("/crm/clients/{client_id}", response_model=CRMClient)
async def get_crm_client(
        client_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        client = db.crm_clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        return CRMClient(
            id=str(client["_id"]),
            clientName=client["clientName"],
            contactInfo=client["contactInfo"],
            serviceBooked=client["serviceBooked"],
            status=client["status"],
            assignedVendor=client.get("assignedVendor"),
            notes=client.get("notes"),
            dueDate=client.get("dueDate"),
            createdAt=client["createdAt"],
            updatedAt=client["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting CRM client: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found",
        )

@app.put("/crm/clients/{client_id}", response_model=CRMClient)
async def update_crm_client(
        client_id: str,
        client_update: CRMClientUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        client = db.crm_clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        update_data = client_update.dict(exclude_unset=True)
        if update_data:
            if "clientName" in update_data and not update_data["clientName"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Client name cannot be empty"
                )
            if "serviceBooked" in update_data:
                service = db.services.find_one({"_id": ObjectId(update_data["serviceBooked"])})
                if not service:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Service not found"
                    )
            if "contactInfo" in update_data and not (update_data["contactInfo"].get("email") or update_data["contactInfo"].get("phone")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="At least one contact method (email or phone) is required"
                )
            update_data["updatedAt"] = datetime.utcnow()
            result = db.crm_clients.update_one(
                {"_id": ObjectId(client_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Client update failed",
                )


        updated_client = db.crm_clients.find_one({"_id": ObjectId(client_id)})

        return CRMClient(
            id=str(updated_client["_id"]),
            clientName=updated_client["clientName"],
            contactInfo=updated_client["contactInfo"],
            serviceBooked=updated_client["serviceBooked"],
            status=updated_client["status"],
            assignedVendor=updated_client.get("assignedVendor"),
            notes=updated_client.get("notes"),
            dueDate=updated_client.get("dueDate"),
            createdAt=updated_client["createdAt"],
            updatedAt=updated_client["updatedAt"]
        )
    except Exception as e:
            logger.error(f"Error updating CRM client: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client update failed",
            )

@app.delete("/crm/clients/{client_id}")
async def delete_crm_client(
        client_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        client = db.crm_clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        result = db.crm_clients.delete_one({"_id": ObjectId(client_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client deletion failed",
            )

        return {"message": "Client deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting CRM client: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client deletion failed",
        )


# Airtable
# @app.post("/bookings/airtable", response_model=Booking)
# async def create_airtable_booking(
#         booking_form: AirtableBookingForm
# ):
#     try:
#         if not booking_form.clientName.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Client name is required"
#             )
#
#         service = db.services.find_one({"_id": ObjectId(booking_form.serviceId)})
#         if not service:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Service not found",
#             )
#
#         if not service["isAvailable"]:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Service is not available",
#             )
#
#         user = db.users.find_one({"email": booking_form.email})
#         if not user:
#             user_dict = {
#                 "email": booking_form.email,
#                 "firstName": booking_form.clientName.split(" ")[0],
#                 "lastName": " ".join(booking_form.clientName.split(" ")[1:]) if len(booking_form.clientName.split(" ")) > 1 else "",
#                 "phone": booking_form.phone,
#                 "role": "user",
#                 "createdAt": datetime.utcnow(),
#                 "updatedAt": datetime.utcnow(),
#                 "hashed_password": get_password_hash(str(uuid.uuid4()))
#             }
#             user_result = db.users.insert_one(user_dict)
#             user_id = str(user_result.inserted_id)
#         else:
#             user_id = str(user["_id"])
#
#         booking_data = BookingInDB(
#             userId=user_id,
#             serviceId=booking_form.serviceId,
#             bookingDate=booking_form.bookingDate,
#             status="pending",
#             specialRequests=booking_form.specialRequests
#         )
#         result = db.bookings.insert_one(booking_data.dict(by_alias=True))
#
#         created_booking = db.bookings.find_one({"_id": result.inserted_id})
#
#         airtable_data = {
#             "Client Name": booking_form.clientName,
#             "Email": booking_form.email,
#             "Phone": booking_form.phone or "",
#             "Service": service["name"],
#             "Booking Date": booking_form.bookingDate.isoformat(),
#             "Special Requests": booking_form.specialRequests or "",
#             "Status": "Pending",
#             "Booking ID": str(created_booking["_id"])
#         }
#         add_to_airtable(airtable_data)
#
#         crm_client = CRMClientInDB(
#             clientName=booking_form.clientName,
#             contactInfo={"email": booking_form.email, "phone": booking_form.phone or ""},
#             serviceBooked=booking_form.serviceId,
#             status="pending"
#         )
#         db.crm_clients.insert_one(crm_client.dict(by_alias=True))
#
#         notification_message = f"""
#         New booking received:
#         - Client: {booking_form.clientName}
#         - Service: {service["name"]}
#         - Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}
#         - Email: {booking_form.email}
#         """
#         send_admin_notification("New Booking Received", notification_message)
#
#         confirmation_html = f"""
#         <html>
#             <body>
#                 <h1>Booking Confirmation</h1>
#                 <p>Dear {booking_form.clientName},</p>
#                 <p>Your booking for {service["name"]} has been received and is currently pending.</p>
#                 <p>Booking Details:</p>
#                 <ul>
#                     <li>Service: {service["name"]}</li>
#                     <li>Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}</li>
#                     <li>Status: Pending</li>
#                     <li>Price: ₦{service["price"]}</li>
#                 </ul>
#                 <p>We will contact you shortly to confirm your booking.</p>
#                 <p>Best regards,<br>The Naija Concierge Team</p>
#             </body>
#         </html>
#         """
#         send_email(booking_form.email, "Booking Confirmation - Naija Concierge", confirmation_html)
#
#         service_obj = Service(
#             id=str(service["_id"]),
#             name=service["name"],
#             description=service["description"],
#             category=service["category"],
#             price=service["price"],
#             image=service.get("image"),
#             duration=service["duration"],
#             isAvailable=service["isAvailable"],
#             createdAt=service["createdAt"],
#             updatedAt=service["updatedAt"]
#         )
#
#         return Booking(
#             id=str(created_booking["_id"]),
#             userId=created_booking["userId"],
#             serviceId=created_booking["serviceId"],
#             bookingDate=created_booking["bookingDate"],
#             status=created_booking["status"],
#             specialRequests=created_booking.get("specialRequests"),
#             createdAt=created_booking["createdAt"],
#             updatedAt=created_booking["updatedAt"],
#             service=service_obj
#         )
#     except Exception as e:
#         logger.error(f"Error creating Airtable booking: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create booking"
#         )


@app.post("/bookings/airtable", response_model=Booking)
async def create_airtable_booking(
    booking_form: AirtableBookingForm,
    current_user: UserInDB = Depends(get_current_active_user)
):
    print(current_user)
    try:
        # Validate client name (not just "string" or empty)
        if not booking_form.clientName.strip() or booking_form.clientName.lower() == "string":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client name must be a valid name, not 'string' or empty"
            )

        # Validate phone (basic format check)
        # if booking_form.phone and not re.match(r"^\+?\d{10,15}$", booking_form.phone):
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="Phone number must be a valid number (10-15 digits, optional leading +)"
        #     )

        # Validate service
        service = db.services.find_one({"_id": ObjectId(booking_form.serviceId)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )

        if not service["isAvailable"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Service is not available"
            )

        # Validate service price
        # if not isinstance(service.get("price"), (int, float)) or service["price"] <= 0:
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="Service price must be a positive number"
        #     )
        #
        # # Validate service name
        # if not service["name"] or service["name"].lower() == "string":
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="Service name is invalid"
        #     )

        # Use current user's email
        booking_email = current_user.email

        # Check if user exists, create if not
        user = db.users.find_one({"email": booking_email})
        if not user:
            user_dict = {
                "email": booking_email,
                "firstName": booking_form.clientName.split(" ")[0],
                "lastName": " ".join(booking_form.clientName.split(" ")[1:]) if len(booking_form.clientName.split(" ")) > 1 else "",
                "phone": booking_form.phone,
                "role": "user",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "hashed_password": get_password_hash(str(uuid.uuid4()))
            }
            user_result = db.users.insert_one(user_dict)
            user_id = str(user_result.inserted_id)
        else:
            user_id = str(user["_id"])

        # Create booking in MongoDB
        booking_data = BookingInDB(
            userId=user_id,
            serviceId=booking_form.serviceId,
            bookingDate=booking_form.bookingDate,
            status="pending",
            specialRequests=booking_form.specialRequests
        )
        result = db.bookings.insert_one(booking_data.dict(by_alias=True))
        created_booking = db.bookings.find_one({"_id": result.inserted_id})

        # Prepare Airtable data (aligned with Airtable schema)
        airtable_data = {
            "Booking ID": str(created_booking["_id"]),  # Explicitly convert ObjectId to string
            "Client Name": booking_form.clientName,
            "Service Requested": service["name"],
            "Booking Date": booking_form.bookingDate.strftime("%Y-%m-%d"),
            "Booking Details": booking_form.specialRequests or "",
            "Status": "Pending",
            "Total Cost": float(service["price"]),  # Ensure number type
            "Feedback": [],  # Optional linked field
            "Subscription Plan": [],  # Optional linked field
            "User": []  # Optional linked field
        }
        logger.info(f"Airtable data prepared: {airtable_data}")

        # Add to Airtable
        airtable_response = add_to_airtable(airtable_data)
        logger.info(f"Airtable response: {airtable_response}")

        # Create CRM client entry
        crm_client = CRMClientInDB(
            clientName=booking_form.clientName,
            contactInfo={"email": booking_email, "phone": booking_form.phone or ""},
            serviceBooked=booking_form.serviceId,
            status="pending"
        )
        db.crm_clients.insert_one(crm_client.dict(by_alias=True))

        # Get user for notification
        user_obj = get_user_by_id(user_id)
        if not user_obj:
            logger.error(f"User not found after creation: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User creation failed"
            )

        # Send admin notification
        notification_message = f"""
        New booking received:
        - Client: {booking_form.clientName}
        - Service: {service["name"]}
        - Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}
        - Email: {booking_email}
        """
        send_admin_notification("New Booking Received", notification_message)

        # Send confirmation email to user
        confirmation_html = f"""
        <html>
            <body>
                <h1>Booking Confirmation</h1>
                <p>Dear {booking_form.clientName},</p>
                <p>Your booking for {service["name"]} has been received and is currently pending.</p>
                <p>Booking Details:</p>
                <ul>
                    <li>Service: {service["name"]}</li>
                    <li>Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}</li>
                    <li>Status: Pending</li>
                    <li>Price: ₦{service["price"]}</li>
                </ul>
                <p>We will contact you shortly to confirm your booking.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(booking_email, "Booking Confirmation - Naija Concierge", confirmation_html)

        # Create service object for response
        service_obj = Service(
            id=str(service["_id"]),
            name=service["name"],
            description=service["description"],
            category=service["category"],
            price=service["price"],
            image=service.get("image"),
            duration=service["duration"],
            isAvailable=service["isAvailable"],
            createdAt=service["createdAt"],
            updatedAt=service["updatedAt"]
        )

        # Return booking response
        return Booking(
            id=str(created_booking["_id"]),
            userId=created_booking["userId"],
            serviceId=created_booking["serviceId"],
            bookingDate=created_booking["bookingDate"],
            status=created_booking["status"],
            specialRequests=created_booking.get("specialRequests"),
            createdAt=created_booking["createdAt"],
            updatedAt=created_booking["updatedAt"],
            service=service_obj
        )
    except Exception as e:
        logger.error(f"Error creating Airtable booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create booking: {str(e)}"
        )
# Service routes
@app.get("/services", response_model=List[Service])
async def get_services(
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None
):
    query = {}
    if category:
        query["category"] = category

    services = list(db.services.find(query).skip(skip).limit(limit))
    return [
        Service(
            id=str(service["_id"]),
            name=service["name"],
            description=service["description"],
            category=service["category"],
            price=service["price"],
            image=service.get("image"),
            duration=service["duration"],
            isAvailable=service["isAvailable"],
            createdAt=service["createdAt"],
            updatedAt=service["updatedAt"]
        ) for service in services
    ]


@app.get("/services/{service_id}", response_model=Service)
async def get_service(service_id: str):
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        return Service(
            id=str(service["_id"]),
            name=service["name"],
            description=service["description"],
            category=service["category"],
            price=service["price"],
            image=service.get("image"),
            duration=service["duration"],
            isAvailable=service["isAvailable"],
            createdAt=service["createdAt"],
            updatedAt=service["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting service: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found",
        )


# @app.post("/services", response_model=Service)
# async def create_service(
#         service: ServiceCreate,
#         current_user: UserInDB = Depends(get_admin_user)
# ):
#     if not service.name.strip() or not service.description.strip() or not service.category.strip() or not service.duration.strip():
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Name, description, category, and duration are required"
#         )
#
#     service_in_db = ServiceInDB(**service.dict())
#     result = db.services.insert_one(service_in_db.dict(by_alias=True))
#
#     created_service = db.services.find_one({"_id": result.inserted_id})
#
#     return Service(
#         id=str(created_service["_id"]),
#         name=created_service["name"],
#         description=created_service["description"],
#         category=created_service["category"],
#         price=created_service["price"],
#         image=created_service.get("image"),
#         duration=created_service["duration"],
#         isAvailable=created_service["isAvailable"],
#         createdAt=created_service["createdAt"],
#         updatedAt=created_service["updatedAt"]
#     )


@app.post("/services", response_model=Service)
async def create_service(
        service: ServiceCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    if not service.name.strip() or not service.description.strip() or not service.category.strip() or not service.duration.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name, description, category, and duration are required"
        )

    service_in_db = ServiceInDB(**service.dict())
    service_dict = service_in_db.dict(by_alias=True)
    if service_dict["_id"] is None:
        del service_dict["_id"]  # Remove _id if it's None to let MongoDB generate it
    result = db.services.insert_one(service_dict)

    created_service = db.services.find_one({"_id": result.inserted_id})

    return Service(
        id=str(created_service["_id"]),
        name=created_service["name"],
        description=created_service["description"],
        category=created_service["category"],
        price=created_service["price"],
        image=created_service.get("image"),
        duration=created_service["duration"],
        isAvailable=created_service["isAvailable"],
        createdAt=created_service["createdAt"],
        updatedAt=created_service["updatedAt"]
    )

@app.put("/services/{service_id}", response_model=Service)
async def update_service(
        service_id: str,
        service_update: ServiceUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        update_data = service_update.dict(exclude_unset=True)
        if update_data:
            if "name" in update_data and not update_data["name"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Name cannot be empty"
                )
            if "description" in update_data and not update_data["description"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Description cannot be empty"
                )
            if "category" in update_data and not update_data["category"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Category cannot be empty"
                )
            if "duration" in update_data and not update_data["duration"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Duration cannot be empty"
                )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.services.update_one(
                {"_id": ObjectId(service_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service update failed",
                )

        updated_service = db.services.find_one({"_id": ObjectId(service_id)})

        return Service(
            id=str(updated_service["_id"]),
            name=updated_service["name"],
            description=updated_service["description"],
            category=updated_service["category"],
            price=updated_service["price"],
            image=updated_service.get("image"),
            duration=updated_service["duration"],
            isAvailable=updated_service["isAvailable"],
            createdAt=updated_service["createdAt"],
            updatedAt=updated_service["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating service: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Service update failed",
        )



@app.delete("/services/{service_id}")
async def delete_service(
        service_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        bookings = db.bookings.find_one({"serviceId": service_id})
        if bookings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete service with existing bookings",
            )

        result = db.services.delete_one({"_id": ObjectId(service_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Service deletion failed",
            )

        return {"message": "Service deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting service: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Service deletion failed",
        )


@app.post("/services/image")
async def upload_service_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/services")

    return {"imageUrl": image_url}


# Package routes
@app.get("/packages", response_model=List[Package])
async def get_packages(
        skip: int = 0,
        limit: int = 100,
        type: Optional[str] = None
):
    query = {}
    if type:
        query["type"] = type

    packages = list(db.packages.find(query).skip(skip).limit(limit))
    return [
        Package(
            id=str(package["_id"]),
            name=package["name"],
            description=package["description"],
            price=package["price"],
            duration=package["duration"],
            features=package["features"],
            image=package.get("image"),
            type=package["type"],
            isPopular=package["isPopular"],
            createdAt=package["createdAt"],
            updatedAt=package["updatedAt"]
        ) for package in packages
    ]


@app.get("/packages/{package_id}", response_model=Package)
async def get_package(package_id: str):
    try:
        package = db.packages.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )

        return Package(
            id=str(package["_id"]),
            name=package["name"],
            description=package["description"],
            price=package["price"],
            duration=package["duration"],
            features=package["features"],
            image=package.get("image"),
            type=package["type"],
            isPopular=package["isPopular"],
            createdAt=package["createdAt"],
            updatedAt=package["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting package: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Package not found",
        )


@app.post("/packages", response_model=Package)
async def create_package(
        package: PackageCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    if not package.name.strip() or not package.description.strip() or not package.type.strip() or not package.duration.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name, description, type, and duration are required"
        )
    if not package.features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one feature is required"
        )

    package_in_db = PackageInDB(**package.dict())
    result = db.packages.insert_one(package_in_db.dict(by_alias=True))

    created_package = db.packages.find_one({"_id": result.inserted_id})

    return Package(
        id=str(created_package["_id"]),
        name=created_package["name"],
        description=created_package["description"],
        price=created_package["price"],
        duration=created_package["duration"],
        features=created_package["features"],
        image=created_package.get("image"),
        type=created_package["type"],
        isPopular=created_package["isPopular"],
        createdAt=created_package["createdAt"],
        updatedAt=created_package["updatedAt"]
    )

@app.put("/packages/{package_id}", response_model=Package)
async def update_package(
        package_id: str,
        package_update: PackageUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        package = db.packages.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )

        update_data = package_update.dict(exclude_unset=True)
        if update_data:
            if "name" in update_data and not update_data["name"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Name cannot be empty"
                )
            if "description" in update_data and not update_data["description"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Description cannot be empty"
                )
            if "type" in update_data and not update_data["type"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Type cannot be empty"
                )
            if "duration" in update_data and not update_data["duration"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Duration cannot be empty"
                )
            if "features" in update_data and not update_data["features"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="At least one feature is required"
                )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.packages.update_one(
                {"_id": ObjectId(package_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Package update failed",
                )

        updated_package = db.packages.find_one({"_id": ObjectId(package_id)})

        return Package(
            id=str(updated_package["_id"]),
            name=updated_package["name"],
            description=updated_package["description"],
            price=updated_package["price"],
            duration=updated_package["duration"],
            features=updated_package["features"],
            image=updated_package.get("image"),
            type=updated_package["type"],
            isPopular=updated_package["isPopular"],
            createdAt=updated_package["createdAt"],
            updatedAt=updated_package["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating package: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Package update failed",
        )


@app.delete("/packages/{package_id}")
async def delete_package(
        package_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        package = db.packages.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )

        subscriptions = db.subscriptions.find_one({"packageId": package_id})
        if subscriptions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete package with existing subscriptions",
            )

        result = db.packages.delete_one({"_id": ObjectId(package_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Package deletion failed",
            )

        return {"message": "Package deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting package: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Package deletion failed",
        )

@app.post("/packages/image")
async def upload_package_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/packages")

    return {"imageUrl": image_url}


# Booking routes
@app.get("/bookings", response_model=List[Booking])
async def get_bookings(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)
    if status:
        query["status"] = status

    bookings = list(db.bookings.find(query).skip(skip).limit(limit))
    result = []

    for booking in bookings:
        try:
            service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
            service_obj = None
            if service:
                service_obj = Service(
                    id=str(service["_id"]),
                    name=service["name"],
                    description=service["description"],
                    category=service["category"],
                    price=service["price"],
                    image=service.get("image"),
                    duration=service["duration"],
                    isAvailable=service["isAvailable"],
                    createdAt=service["createdAt"],
                    updatedAt=service["updatedAt"]
                )

            result.append(
                Booking(
                    id=str(booking["_id"]),
                    userId=booking["userId"],
                    serviceId=booking["serviceId"],
                    bookingDate=booking["bookingDate"],
                    status=booking["status"],
                    specialRequests=booking.get("specialRequests"),
                    createdAt=booking["createdAt"],
                    updatedAt=booking["updatedAt"],
                    service=service_obj
                )
            )
        except Exception as e:
            logger.error(f"Error processing booking: {e}")

    return result


@app.get("/bookings/{booking_id}", response_model=Booking)
async def get_booking(
        booking_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found",
            )

        # Regular users can only see their own bookings
        if current_user.role != "admin" and booking["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
        service_obj = None
        if service:
            service_obj = Service(
                id=str(service["_id"]),
                name=service["name"],
                description=service["description"],
                category=service["category"],
                price=service["price"],
                image=service.get("image"),
                duration=service["duration"],
                isAvailable=service["isAvailable"],
                createdAt=service["createdAt"],
                updatedAt=service["updatedAt"]
            )

        return Booking(
            id=str(booking["_id"]),
            userId=booking["userId"],
            serviceId=booking["serviceId"],
            bookingDate=booking["bookingDate"],
            status=booking["status"],
            specialRequests=booking.get("specialRequests"),
            createdAt=booking["createdAt"],
            updatedAt=booking["updatedAt"],
            service=service_obj
        )
    except Exception as e:
        logger.error(f"Error getting booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Booking not found",
        )


@app.post("/bookings", response_model=Booking)
async def create_booking(
        booking: BookingCreate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.role != "admin" and booking.userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    try:
        service = db.services.find_one({"_id": ObjectId(booking.serviceId)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        if not service["isAvailable"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Service is not available",
            )
    except Exception as e:
        logger.error(f"Error checking service: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid service ID",
        )

    booking_in_db = BookingInDB(**booking.dict())
    result = db.bookings.insert_one(booking_in_db.dict(by_alias=True))

    created_booking = db.bookings.find_one({"_id": result.inserted_id})

    service_obj = Service(
        id=str(service["_id"]),
        name=service["name"],
        description=service["description"],
        category=service["category"],
        price=service["price"],
        image=service.get("image"),
        duration=service["duration"],
        isAvailable=service["isAvailable"],
        createdAt=service["createdAt"],
        updatedAt=service["updatedAt"]
    )

    user = get_user_by_id(booking.userId)
    if user:
        booking_html = f"""
        <html>
            <body>
                <h1>Booking Confirmation</h1>
                <p>Dear {user.firstName},</p>
                <p>Your booking for {service["name"]} has been received and is currently {created_booking["status"]}.</p>
                <p>Booking Details:</p>
                <ul>
                    <li>Service: {service["name"]}</li>
                    <li>Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}</li>
                    <li>Status: {created_booking["status"]}</li>
                    <li>Price: ₦{service["price"]}</li>
                </ul>
                <p>We will contact you shortly to confirm your booking.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(user.email, "Booking Confirmation - Naija Concierge", booking_html)

    notification_message = f"""
    New booking created:
    - Client: {user.firstName} {user.lastName}
    - Service: {service["name"]}
    - Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}
    - Status: {created_booking["status"]}
    """
    send_admin_notification("New Booking Created", notification_message)

    return Booking(
        id=str(created_booking["_id"]),
        userId=created_booking["userId"],
        serviceId=created_booking["serviceId"],
        bookingDate=created_booking["bookingDate"],
        status=created_booking["status"],
        specialRequests=created_booking.get("specialRequests"),
        createdAt=created_booking["createdAt"],
        updatedAt=created_booking["updatedAt"],
        service=service_obj
    )


@app.put("/bookings/{booking_id}", response_model=Booking)
async def update_booking(
        booking_id: str,
        booking_update: BookingUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found",
            )

        if current_user.role != "admin" and booking["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        update_data = booking_update.dict(exclude_unset=True)
        if current_user.role != "admin" and "status" in update_data:
            if update_data["status"] != "cancelled":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to change status",
                )

        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            result = db.bookings.update_one(
                {"_id": ObjectId(booking_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Booking update failed",
                )

        updated_booking = db.bookings.find_one({"_id": ObjectId(booking_id)})

        service = db.services.find_one({"_id": ObjectId(updated_booking["serviceId"])})
        service_obj = None
        if service:
            service_obj = Service(
                id=str(service["_id"]),
                name=service["name"],
                description=service["description"],
                category=service["category"],
                price=service["price"],
                image=service.get("image"),
                duration=service["duration"],
                isAvailable=service["isAvailable"],
                createdAt=service["createdAt"],
                updatedAt=service["updatedAt"]
            )

        if "status" in update_data:
            user = get_user_by_id(updated_booking["userId"])
            if user:
                status_update_html = f"""
                <html>
                    <body>
                        <h1>Booking Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your booking for {service["name"]} has been updated to {updated_booking["status"]}.</p>
                        <p>Booking Details:</p>
                        <ul>
                            <li>Service: {service["name"]}</li>
                            <li>Date: {updated_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}</li>
                            <li>Status: {updated_booking["status"]}</li>
                        </ul>
                        <p>If you have any questions, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Booking Status Update - Naija Concierge", status_update_html)

            notification_message = f"""
            Booking status updated:
            - Client: {user.firstName} {user.lastName}
            - Service: {service["name"]}
            - Date: {updated_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}
            - New Status: {updated_booking["status"]}
            """
            send_admin_notification("Booking Status Updated", notification_message)

        return Booking(
            id=str(updated_booking["_id"]),
            userId=updated_booking["userId"],
            serviceId=updated_booking["serviceId"],
            bookingDate=updated_booking["bookingDate"],
            status=updated_booking["status"],
            specialRequests=updated_booking.get("specialRequests"),
            createdAt=updated_booking["createdAt"],
            updatedAt=updated_booking["updatedAt"],
            service=service_obj
        )
    except Exception as e:
        logger.error(f"Error updating booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Booking update failed",
        )

@app.delete("/bookings/{booking_id}")
async def delete_booking(
        booking_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found",
            )

        result = db.bookings.delete_one({"_id": ObjectId(booking_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Booking deletion failed",
            )

        return {"message": "Booking deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Booking deletion failed",
        )

# Subscription routes
@app.get("/subscriptions", response_model=List[Subscription])
async def get_subscriptions(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)
    if status:
        query["status"] = status

    subscriptions = list(db.subscriptions.find(query).skip(skip).limit(limit))
    result = []

    for subscription in subscriptions:
        try:
            package = db.packages.find_one({"_id": ObjectId(subscription["packageId"])})
            package_obj = None
            if package:
                package_obj = Package(
                    id=str(package["_id"]),
                    name=package["name"],
                    description=package["description"],
                    price=package["price"],
                    duration=package["duration"],
                    features=package["features"],
                    image=package.get("image"),
                    type=package["type"],
                    isPopular=package["isPopular"],
                    createdAt=package["createdAt"],
                    updatedAt=package["updatedAt"]
                )

            result.append(
                Subscription(
                    id=str(subscription["_id"]),
                    userId=subscription["userId"],
                    packageId=subscription["packageId"],
                    startDate=subscription["startDate"],
                    endDate=subscription["endDate"],
                    status=subscription["status"],
                    createdAt=subscription["createdAt"],
                    updatedAt=subscription["updatedAt"],
                    package=package_obj
                )
            )
        except Exception as e:
            logger.error(f"Error processing subscription: {e}")

    return result

@app.get("/subscriptions/{subscription_id}", response_model=Subscription)
async def get_subscription(
        subscription_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )

        if current_user.role != "admin" and subscription["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        package = db.packages.find_one({"_id": ObjectId(subscription["packageId"])})
        package_obj = None
        if package:
            package_obj = Package(
                id=str(package["_id"]),
                name=package["name"],
                description=package["description"],
                price=package["price"],
                duration=package["duration"],
                features=package["features"],
                image=package.get("image"),
                type=package["type"],
                isPopular=package["isPopular"],
                createdAt=package["createdAt"],
                updatedAt=package["updatedAt"]
            )

        return Subscription(
            id=str(subscription["_id"]),
            userId=subscription["userId"],
            packageId=subscription["packageId"],
            startDate=subscription["startDate"],
            endDate=subscription["endDate"],
            status=subscription["status"],
            createdAt=subscription["createdAt"],
            updatedAt=subscription["updatedAt"],
            package=package_obj
        )
    except Exception as e:
        logger.error(f"Error getting subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        )


# @app.post("/subscriptions", response_model=Subscription)
# async def create_subscription(
#         subscription: SubscriptionCreate,
#         current_user: UserInDB = Depends(get_current_active_user)
# ):
#     if current_user.role != "admin" and subscription.userId != str(current_user.id):
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Not enough permissions",
#         )
#
#     try:
#         package = db.packages.find_one({"_id": ObjectId(subscription.packageId)})
#         if not package:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Package not found",
#             )
#     except Exception as e:
#         logger.error(f"Error checking package: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Invalid package ID",
#         )
#
#     active_subscription = db.subscriptions.find_one({
#         "userId": subscription.userId,
#         "status": "active"
#     })
#     if active_subscription:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="User already has an active subscription",
#         )
#
#     subscription_in_db = SubscriptionInDB(**subscription.dict())
#     result = db.subscriptions.insert_one(subscription_in_db.dict(by_alias=True))
#
#     created_subscription = db.subscriptions.find_one({"_id": result.inserted_id})
#
#     package_obj = Package(
#         id=str(package["_id"]),
#         name=package["name"],
#         description=package["description"],
#         price=package["price"],
#         duration=package["duration"],
#         features=package["features"],
#         image=package.get("image"),
#         type=package["type"],
#         isPopular=package["isPopular"],
#         createdAt=package["createdAt"],
#         updatedAt=package["updatedAt"]
#     )
#
#     user = get_user_by_id(subscription.userId)
#     if user:
#         subscription_html = f"""
#         <html>
#             <body>
#                 <h1>Subscription Confirmation</h1>
#                 <p>Dear {user.firstName},</p>
#                 <p>Your subscription to the {package["name"]} package has been successfully created.</p>
#                 <p>Subscription Details:</p>
#                 <ul>
#                     <li>Package: {package["name"]}</li>
#                     <li>Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}</li>
#                     <li>End Date: {created_subscription["endDate"].strftime("%Y-%m-%d")}</li>
#                     <li>Price: ₦{package["price"]}</li>
#                     <li>Status: {created_subscription["status"]}</li>
#                 </ul>
#                 <p>Thank you for choosing Naija Concierge. If you have any questions, please contact us.</p>
#                 <p>Best regards,<br>The Naija Concierge Team</p>
#             </body>
#         </html>
#         """
#         send_email(user.email, "Subscription Confirmation - Naija Concierge", subscription_html)
#
#     notification_message = f"""
#     New subscription created:
#     - Client: {user.firstName} {user.lastName}
#     - Package: {package["name"]}
#     - Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}
#     - Status: {created_subscription["status"]}
#     """
#     send_admin_notification("New Subscription Created", notification_message)
#
#     return Subscription(
#         id=str(created_subscription["_id"]),
#         userId=created_subscription["userId"],
#         packageId=created_subscription["packageId"],
#         startDate=created_subscription["startDate"],
#         endDate=created_subscription["endDate"],
#         status=created_subscription["status"],
#         createdAt=created_subscription["createdAt"],
#         updatedAt=created_subscription["updatedAt"],
#         package=package_obj
#     )



@app.post("/subscriptions", response_model=Subscription)
async def create_subscription(
        subscription: SubscriptionCreate,
        current_user: dict = Depends(get_current_active_user)
):
    if current_user["role"] != "admin" and subscription.userId != str(current_user["id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    try:
        package = db.packages.find_one({"_id": ObjectId(subscription.packageId)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )
    except Exception as e:
        logger.error(f"Error checking package: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid package ID",
        )

    active_subscription = db.subscriptions.find_one({
        "userId": subscription.userId,
        "status": "active"
    })
    if active_subscription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has an active subscription",
        )

    # Verify payment transaction
    transaction = db.transactions.find_one({
        "subscriptionId": f"pending_{subscription.userId}_{subscription.packageId}",
        "status": "success"
    })
    if not transaction:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Payment required to create subscription",
        )

    # Create subscription
    subscription_in_db = SubscriptionInDB(
        userId=subscription.userId,
        packageId=subscription.packageId,
        startDate=datetime.utcnow(),
        endDate=datetime.utcnow() + timedelta(days=package["duration"]),
        status=SubscriptionStatus.active,
        createdAt=datetime.utcnow(),
        updatedAt=datetime.utcnow()
    )
    result = db.subscriptions.insert_one(subscription_in_db.dict(by_alias=True))

    created_subscription = db.subscriptions.find_one({"_id": result.inserted_id})

    package_obj = Package(
        id=str(package["_id"]),
        name=package["name"],
        description=package["description"],
        price=package["price"],
        duration=package["duration"],
        features=package["features"],
        image=package.get("image"),
        type=package["type"],
        isPopular=package["isPopular"],
        createdAt=package["createdAt"],
        updatedAt=package["updatedAt"]
    )

    user = get_user_by_id(subscription.userId)
    if user:
        subscription_html = f"""
        <html>
            <body>
                <h1>Subscription Confirmation</h1>
                <p>Dear {user["firstName"]},</p>
                <p>Your subscription to the {package["name"]} package has been successfully created.</p>
                <p>Subscription Details:</p>
                <ul>
                    <li>Package: {package["name"]}</li>
                    <li>Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}</li>
                    <li>End Date: {created_subscription["endDate"].strftime("%Y-%m-%d")}</li>
                    <li>Price: {transaction["currency"]} {transaction["amount"]}</li>
                    <li>Status: {created_subscription["status"]}</li>
                </ul>
                <p>Thank you for choosing Naija Concierge. If you have any questions, please contact us.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(user["email"], "Subscription Confirmation - Naija Concierge", subscription_html)

    notification_message = f"""
    New subscription created:
    - Client: {user["firstName"]} {user["lastName"]}
    - Package: {package["name"]}
    - Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}
    - Status: {created_subscription["status"]}
    - Amount Paid: {transaction["currency"]} {transaction["amount"]}
    """
    send_admin_notification("New Subscription Created", notification_message)

    return Subscription(
        id=str(created_subscription["_id"]),
        userId=created_subscription["userId"],
        packageId=created_subscription["packageId"],
        startDate=created_subscription["startDate"],
        endDate=created_subscription["endDate"],
        status=created_subscription["status"],
        createdAt=created_subscription["createdAt"],
        updatedAt=created_subscription["updatedAt"],
        package=package_obj
    )


@app.post("/subscriptions/initiate_payment")
async def initiate_subscription_payment(
    subscription: SubscriptionInitiate,
    current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.role != "admin" and subscription.userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    try:
        package = db.packages.find_one({"_id": ObjectId(subscription.packageId)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )
    except Exception as e:
        logger.error(f"Error checking package: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid package ID",
        )

    # Initiate Flutterwave payment
    tx_ref = f"sub_init_{subscription.userId}_{subscription.packageId}_{int(datetime.utcnow().timestamp())}"
    payment_response = await initialize_flutterwave_payment(
        email=current_user.email,
        amount=package["price"],
        currency="NGN",
        tx_ref=tx_ref,
        country="NG"
    )

    # Log the full response for debugging
    logger.info(f"Flutterwave payment response: {payment_response}")

    # Check if response has expected structure
    if payment_response.get("status") != "success" or "data" not in payment_response:
        logger.error(f"Unexpected Flutterwave response: {payment_response}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to initiate payment with Flutterwave",
        )

    # Extract transaction ID and payment link
    data = payment_response["data"]
    transaction_id = data.get("tx_ref") or data.get("transaction_id") or tx_ref  # Fallback to tx_ref if no ID
    payment_link = data.get("link")

    if not payment_link:
        logger.error(f"No payment link in Flutterwave response: {payment_response}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment link not provided by Flutterwave",
        )

    # Store pending transaction
    transaction = TransactionInDB(
        tx_ref=tx_ref,
        transactionId=transaction_id,  # Use the extracted transaction ID
        userId=subscription.userId,
        packageId=subscription.packageId,
        amount=package["price"],
        currency="NGN",
        preferredCurrency=subscription.preferredCurrency,
        status="pending",
        createdAt=datetime.utcnow(),
        updatedAt=datetime.utcnow()
    )
    db.transactions.insert_one(transaction.dict(by_alias=True))

    return {"payment_url": payment_link}

@app.post("/webhooks/flutterwave")
async def handle_flutterwave_webhook(
        payload: dict,
        request: Request
):
    signature = request.headers.get("verif-hash")
    if not signature or not verify_webhook_signature(await request.body(), signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = payload.get("event")
    data = payload.get("data")

    if event == "charge.completed" and data["status"] == "successful":
        transaction_id = str(data["id"])
        tx_ref = data["tx_ref"]
        transaction = db.transactions.find_one({"tx_ref": tx_ref, "transactionId": transaction_id})
        if not transaction:
            logger.error(f"Transaction not found for tx_ref: {tx_ref}, ID: {transaction_id}")
            return {"status": "error"}

        # Verify payment
        verification = await verify_flutterwave_payment(transaction_id)
        if verification["status"] == "success" and verification["data"]["status"] == "successful":
            # Update transaction status
            db.transactions.update_one(
                {"tx_ref": tx_ref},
                {"$set": {"status": "success", "updatedAt": datetime.utcnow()}}
            )
            logger.info(f"Payment verified for tx_ref: {tx_ref}")

            # Create subscription
            subscription = await create_subscription_from_transaction(transaction)
            if subscription:
                logger.info(f"Subscription created for tx_ref: {tx_ref}")
            else:
                logger.error(f"Failed to create subscription for tx_ref: {tx_ref}")
        else:
            logger.error(f"Payment verification failed for tx_ref: {tx_ref}")
            db.transactions.update_one(
                {"tx_ref": tx_ref},
                {"$set": {"status": "failed", "updatedAt": datetime.utcnow()}}
            )

    return {"status": "success"}

@app.get("/payment/callback")
async def payment_callback(
        status: str,
        tx_ref: Optional[str] = None,
        transaction_id: Optional[str] = None
):
    if status == "successful" and tx_ref:
        transaction = db.transactions.find_one({"tx_ref": tx_ref})
        if transaction and transaction["status"] == "success":
            return {"message": "Payment successful. Subscription created."}
        return {"message": "Payment received. Awaiting confirmation."}
    elif status == "cancelled":
        return {"message": "Payment cancelled. Please try again."}
    else:
        return {"message": "Payment failed. Please contact support."}


@app.put("/subscriptions/{subscription_id}", response_model=Subscription)
async def update_subscription(
        subscription_id: str,
        subscription_update: SubscriptionUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )

        if current_user.role != "admin" and subscription["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        update_data = subscription_update.dict(exclude_unset=True)
        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            result = db.subscriptions.update_one(
                {"_id": ObjectId(subscription_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Subscription update failed",
                )

        updated_subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})

        package = db.packages.find_one({"_id": ObjectId(updated_subscription["packageId"])})
        package_obj = None
        if package:
            package_obj = Package(
                id=str(package["_id"]),
                name=package["name"],
                description=package["description"],
                price=package["price"],
                duration=package["duration"],
                features=package["features"],
                image=package.get("image"),
                type=package["type"],
                isPopular=package["isPopular"],
                createdAt=package["createdAt"],
                updatedAt=package["updatedAt"]
            )

        if "status" in update_data:
            user = get_user_by_id(updated_subscription["userId"])
            if user:
                status_update_html = f"""
                <html>
                    <body>
                        <h1>Subscription Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your subscription to the {package["name"]} package has been updated to {updated_subscription["status"]}.</p>
                        <p>Subscription Details:</p>
                        <ul>
                            <li>Package: {package["name"]}</li>
                            <li>Start Date: {updated_subscription["startDate"].strftime("%Y-%m-%d")}</li>
                            <li>End Date: {updated_subscription["endDate"].strftime("%Y-%m-%d")}</li>
                            <li>Status: {updated_subscription["status"]}</li>
                        </ul>
                        <p>If you have any questions, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Subscription Status Update - Naija Concierge", status_update_html)

            notification_message = f"""
            Subscription status updated:
            - Client: {user.firstName} {user.lastName}
            - Package: {package["name"]}
            - New Status: {updated_subscription["status"]}
            """
            send_admin_notification("Subscription Status Updated", notification_message)

        return Subscription(
            id=str(updated_subscription["_id"]),
            userId=updated_subscription["userId"],
            packageId=updated_subscription["packageId"],
            startDate=updated_subscription["startDate"],
            endDate=updated_subscription["endDate"],
            status=updated_subscription["status"],
            createdAt=updated_subscription["createdAt"],
            updatedAt=updated_subscription["updatedAt"],
            package=package_obj
        )
    except Exception as e:
        logger.error(f"Error updating subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription update failed",
        )

@app.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
        subscription_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )

        result = db.subscriptions.delete_one({"_id": ObjectId(subscription_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subscription deletion failed",
            )

        package = db.packages.find_one({"_id": ObjectId(subscription["packageId"])})
        user = get_user_by_id(subscription["userId"])
        if user and package:
            cancellation_html = f"""
            <html>
                <body>
                    <h1>Subscription Cancelled</h1>
                    <p>Dear {user.firstName},</p>
                    <p>Your subscription to the {package["name"]} package has been cancelled.</p>
                    <p>If you have any questions or would like to reactivate, please contact us.</p>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            send_email(user.email, "Subscription Cancellation - Naija Concierge", cancellation_html)

        return {"message": "Subscription deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription deletion failed",
        )





# Blog routes

@app.get("/blogs", response_model=List[Blog])
async def get_blogs(
        skip: int = 0,
        limit: int = 100,
        tag: Optional[str] = None
):
    query = {}
    if tag:
        query["tags"] = {"$in": [tag]}

    blogs = list(db.blogs.find(query).skip(skip).limit(limit))
    return [
        Blog(
            id=str(blog["_id"]),
            title=blog["title"],
            slug=blog["slug"],
            content=blog["content"],
            excerpt=blog["excerpt"],
            coverImage=blog.get("coverImage"),
            author=blog["author"],
            tags=blog["tags"],
            createdAt=blog["createdAt"],
            updatedAt=blog["updatedAt"]
        ) for blog in blogs
    ]


@app.get("/blogs/{blog_id}", response_model=Blog)
async def get_blog(blog_id: str):
    try:
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        return Blog(
            id=str(blog["_id"]),
            title=blog["title"],
            slug=blog["slug"],
            content=blog["content"],
            excerpt=blog["excerpt"],
            coverImage=blog.get("coverImage"),
            author=blog["author"],
            tags=blog["tags"],
            createdAt=blog["createdAt"],
            updatedAt=blog["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting blog: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found",
        )

@app.get("/blogs/blog/{slug}", response_model=Blog)
async def get_blog_by_slug(slug: str):
    blog = db.blogs.find_one({"slug": slug})
    if not blog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found",
        )

    return Blog(
        id=str(blog["_id"]),
        title=blog["title"],
        slug=blog["slug"],
        content=blog["content"],
        excerpt=blog["excerpt"],
        coverImage=blog.get("coverImage"),
        author=blog["author"],
        tags=blog["tags"],
        createdAt=blog["createdAt"],
        updatedAt=blog["updatedAt"]
    )


@app.post("/blogs", response_model=Blog)
async def create_blog(
        blog: BlogCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    if not blog.title.strip() or not blog.slug.strip() or not blog.content.strip() or not blog.excerpt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title, slug, content, and excerpt are required"
        )
    if not blog.author.get("name"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Author name is required"
        )

    existing_blog = db.blogs.find_one({"slug": blog.slug})
    if existing_blog:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blog with this slug already exists",
        )

    blog_in_db = BlogInDB(**blog.dict())
    result = db.blogs.insert_one(blog_in_db.dict(by_alias=True))

    created_blog = db.blogs.find_one({"_id": result.inserted_id})

    return Blog(
        id=str(created_blog["_id"]),
        title=created_blog["title"],
        slug=created_blog["slug"],
        content=created_blog["content"],
        excerpt=created_blog["excerpt"],
        coverImage=created_blog.get("coverImage"),
        author=created_blog["author"],
        tags=created_blog["tags"],
        createdAt=created_blog["createdAt"],
        updatedAt=created_blog["updatedAt"]
    )



@app.put("/blogs/{blog_id}", response_model=Blog)
async def update_blog(
        blog_id: str,
        blog_update: BlogUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        update_data = blog_update.dict(exclude_unset=True)
        if update_data:
            if "title" in update_data and not update_data["title"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Title cannot be empty"
                )
            if "slug" in update_data:
                if not update_data["slug"].strip():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Slug cannot be empty"
                    )
                existing_blog = db.blogs.find_one({"slug": update_data["slug"], "_id": {"$ne": ObjectId(blog_id)}})
                if existing_blog:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Blog with this slug already exists",
                    )
            if "content" in update_data and not update_data["content"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Content cannot be empty"
                )
            if "excerpt" in update_data and not update_data["excerpt"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Excerpt cannot be empty"
                )
            if "author" in update_data and not update_data["author"].get("name"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Author name is required"
                )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.blogs.update_one(
                {"_id": ObjectId(blog_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Blog update failed",
                )

        updated_blog = db.blogs.find_one({"_id": ObjectId(blog_id)})

        return Blog(
            id=str(updated_blog["_id"]),
            title=updated_blog["title"],
            slug=updated_blog["slug"],
            content=updated_blog["content"],
            excerpt=updated_blog["excerpt"],
            coverImage=updated_blog.get("coverImage"),
            author=updated_blog["author"],
            tags=updated_blog["tags"],
            createdAt=updated_blog["createdAt"],
            updatedAt=updated_blog["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating blog: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blog update failed",
        )

@app.delete("/blogs/{blog_id}")
async def delete_blog(
        blog_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if blog exists
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        # Delete blog
        result = db.blogs.delete_one({"_id": ObjectId(blog_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Blog deletion failed",
            )

        return {"message": "Blog deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting blog: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blog deletion failed",
        )


@app.post("/blogs/image")
async def upload_blog_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/blogs")

    return {"imageUrl": image_url}


# Emergency Alert routes
@app.get("/emergency-alerts", response_model=List[EmergencyAlert])
async def get_emergency_alerts(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}

    # Regular users can only see their own alerts
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)

    if status:
        query["status"] = status

    alerts = list(db.emergency_alerts.find(query).skip(skip).limit(limit))
    return [
        EmergencyAlert(
            id=str(alert["_id"]),
            userId=alert["userId"],
            message=alert["message"],
            location=alert.get("location"),
            status=alert["status"],
            createdAt=alert["createdAt"],
            updatedAt=alert["updatedAt"]
        ) for alert in alerts
    ]


@app.post("/emergency-alerts", response_model=EmergencyAlert)
async def create_emergency_alert(
        alert: EmergencyAlertCreate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    if not alert.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is required"
        )

    if alert.userId != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    alert_in_db = EmergencyAlertInDB(**alert.dict())
    result = db.emergency_alerts.insert_one(alert_in_db.dict(by_alias=True))

    created_alert = db.emergency_alerts.find_one({"_id": result.inserted_id})

    user = get_user_by_id(alert.userId)
    if user:
        alert_html = f"""
        <html>
            <body>
                <h1>Emergency Alert Received</h1>
                <p>Dear {user.firstName},</p>
                <p>We have received your emergency alert: {alert.message}</p>
                <p>Our team is responding and will contact you shortly.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(user.email, "Emergency Alert - Naija Concierge", alert_html)

    notification_message = f"""
    New emergency alert:
    - Client: {user.firstName} {user.lastName}
    - Message: {alert.message}
    - Location: {alert.location or "Not provided"}
    - Status: {created_alert["status"]}
    """
    send_admin_notification("New Emergency Alert", notification_message)

    return EmergencyAlert(
        id=str(created_alert["_id"]),
        userId=created_alert["userId"],
        message=created_alert["message"],
        location=created_alert.get("location"),
        status=created_alert["status"],
        createdAt=created_alert["createdAt"],
        updatedAt=created_alert["updatedAt"]
    )



@app.put("/emergency-alerts/{alert_id}", response_model=EmergencyAlert)
async def update_emergency_alert(
        alert_id: str,
        alert_update: EmergencyAlertUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        alert = db.emergency_alerts.find_one({"_id": ObjectId(alert_id)})
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Emergency alert not found",
            )

        update_data = alert_update.dict(exclude_unset=True)
        if update_data:
            if "message" in update_data and not update_data["message"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message cannot be empty"
                )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.emergency_alerts.update_one(
                {"_id": ObjectId(alert_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Emergency alert update failed",
                )

        updated_alert = db.emergency_alerts.find_one({"_id": ObjectId(alert_id)})

        if "status" in update_data:
            user = get_user_by_id(updated_alert["userId"])
            if user:
                status_update_html = f"""
                <html>
                    <body>
                        <h1>Emergency Alert Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your emergency alert has been updated to {updated_alert["status"]}.</p>
                        <p>Message: {updated_alert["message"]}</p>
                        <p>If you need further assistance, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Emergency Alert Status Update - Naija Concierge", status_update_html)

        return EmergencyAlert(
            id=str(updated_alert["_id"]),
            userId=updated_alert["userId"],
            message=updated_alert["message"],
            location=updated_alert.get("location"),
            status=updated_alert["status"],
            createdAt=updated_alert["createdAt"],
            updatedAt=updated_alert["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating emergency alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Emergency alert update failed",
        )


# Contact routes
# @app.post("/contact")
# async def send_contact_message(message: ContactMessage):
#     try:
#         # Store message in database
#         message_dict = message.dict()
#         message_dict["createdAt"] = datetime.utcnow()
#         db.contact_messages.insert_one(message_dict)
#
#         # Send notification email to admin
#         admin_users = list(db.users.find({"role": "admin"}))
#         for admin in admin_users:
#             contact_html = f"""
#             <html>
#                 <body>
#                     <h1>New Contact Message</h1>
#                     <p><strong>Name:</strong> {message.name}</p>
#                     <p><strong>Email:</strong> {message.email}</p>
#                     <p><strong>Phone:</strong> {message.phone or "Not provided"}</p>
#                     <p><strong>Subject:</strong> {message.subject}</p>
#                     <p><strong>Message:</strong></p>
#                     <p>{message.message}</p>
#                 </body>
#             </html>
#             """
#             send_email(admin["email"], f"New Contact Message: {message.subject}", contact_html)
#
#         # Send confirmation email to user
#         confirmation_html = f"""
#         <html>
#             <body>
#                 <h1>Thank You for Contacting Us</h1>
#                 <p>Dear {message.name},</p>
#                 <p>We have received your message and will get back to you shortly.</p>
#                 <p><strong>Subject:</strong> {message.subject}</p>
#                 <p><strong>Message:</strong></p>
#                 <p>{message.message}</p>
#                 <p>Best regards,<br>The Naija Concierge Team</p>
#             </body>
#         </html>
#         """
#         send_email(message.email, "Thank You for Contacting Naija Concierge", confirmation_html)
#
#         return {"message": "Contact message sent successfully"}
#     except Exception as e:
#         logger.error(f"Error sending contact message: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to send contact message",
#         )


@app.post("/contact")
async def send_contact_message(message: ContactMessage):
    try:
        if not message.name.strip() or not message.subject.strip() or not message.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name, subject, and message are required"
            )

        message_dict = message.dict()
        message_dict["createdAt"] = datetime.utcnow()
        db.contact_messages.insert_one(message_dict)

        contact_html = f"""
        <html>
            <body>
                <h1>New Contact Message</h1>
                <p><strong>Name:</strong> {message.name}</p>
                <p><strong>Email:</strong> {message.email}</p>
                <p><strong>Phone:</strong> {message.phone or "Not provided"}</p>
                <p><strong>Subject:</strong> {message.subject}</p>
                <p><strong>Message:</strong></p>
                <p>{message.message}</p>
            </body>
        </html>
        """
        send_admin_notification(f"New Contact Message: {message.subject}", contact_html)

        confirmation_html = f"""
        <html>
            <body>
                <h1>Thank You for Contacting Us</h1>
                <p>Dear {message.name},</p>
                <p>We have received your message and will get back to you shortly.</p>
                <p><strong>Subject:</strong> {message.subject}</p>
                <p><strong>Message:</strong></p>
                <p>{message.message}</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(message.email, "Thank You for Contacting Naija Concierge", confirmation_html)

        notification_message = f"""
        New contact message received:
        - Name: {message.name}
        - Email: {message.email}
        - Subject: {message.subject}
        """
        send_admin_notification("New Contact Message", notification_message)

        return {"message": "Contact message sent successfully"}
    except Exception as e:
        logger.error(f"Error sending contact message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send contact message",
        )


@app.post("/webhooks/booking-notification")
async def booking_notification_webhook(data: Dict[str, Any]):
    try:
        booking_id = data.get("bookingId")
        if not booking_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Booking ID required"
            )

        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
        user = db.users.find_one({"_id": ObjectId(booking["userId"])})

        if not service or not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service or user not found"
            )

        notification_message = f"""
        New Booking:
        - Client: {user["firstName"]} {user["lastName"]}
        - Service: {service["name"]}
        - Date: {booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}
        - Status: {booking["status"]}
        """

        send_admin_notification("New Booking Notification", notification_message)

        return {"message": "Notification processed successfully"}
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook"
        )


# Document routes
@app.get("/documents", response_model=List[Document])
async def get_documents(
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}

    # Regular users can only see their own documents
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)

    documents = list(db.documents.find(query))
    return [
        Document(
            id=str(doc["_id"]),
            userId=doc["userId"],
            name=doc["name"],
            type=doc["type"],
            url=doc["url"],
            uploadDate=doc["uploadDate"]
        ) for doc in documents
    ]


@app.post("/documents", response_model=Document)
async def create_document(
        name: str = Form(...),
        type: str = Form(...),
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_current_active_user)
):
    if not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document name is required"
        )
    if not type.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document type is required"
        )

    file_url = upload_file_to_cloudinary(file, folder="naija_concierge/documents")

    document_in_db = DocumentInDB(
        userId=str(current_user.id),
        name=name,
        type=type,
        url=file_url
    )
    result = db.documents.insert_one(document_in_db.dict(by_alias=True))

    created_document = db.documents.find_one({"_id": result.inserted_id})

    notification_message = f"""
    New document uploaded:
    - Client: {current_user.firstName} {current_user.lastName}
    - Document: {name}
    - Type: {type}
    """
    send_admin_notification("New Document Uploaded", notification_message)

    return Document(
        id=str(created_document["_id"]),
        userId=created_document["userId"],
        name=created_document["name"],
        type=created_document["type"],
        url=created_document["url"],
        uploadDate=created_document["uploadDate"]
    )

@app.delete("/documents/{document_id}")
async def delete_document(
        document_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        document = db.documents.find_one({"_id": ObjectId(document_id)})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        if current_user.role != "admin" and document["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        result = db.documents.delete_one({"_id": ObjectId(document_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document deletion failed",
            )

        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document deletion failed",
        )


# Admin dashboard stats
@app.get("/admin/stats")
async def get_admin_stats(current_user: UserInDB = Depends(get_admin_user)):
    try:
        # Get total users
        total_users = db.users.count_documents({})

        # Get total bookings
        total_bookings = db.bookings.count_documents({})

        # Get total revenue
        bookings = list(db.bookings.find({"status": {"$in": ["confirmed", "completed"]}}))
        total_revenue = 0
        for booking in bookings:
            try:
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    total_revenue += service["price"]
            except Exception:
                pass

        # Get active packages
        active_packages = db.subscriptions.count_documents({"status": "active"})

        # Get user growth (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        new_users = db.users.count_documents({"createdAt": {"$gte": thirty_days_ago}})
        user_growth = (new_users / total_users) * 100 if total_users > 0 else 0

        # Get booking growth (last 30 days)
        new_bookings = db.bookings.count_documents({"createdAt": {"$gte": thirty_days_ago}})
        booking_growth = (new_bookings / total_bookings) * 100 if total_bookings > 0 else 0

        # Get revenue growth (compare last 30 days with previous 30 days)
        last_30_days_bookings = list(db.bookings.find({
            "createdAt": {"$gte": thirty_days_ago},
            "status": {"$in": ["confirmed", "completed"]}
        }))
        last_30_days_revenue = 0
        for booking in last_30_days_bookings:
            try:
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    last_30_days_revenue += service["price"]
            except Exception:
                pass

        previous_30_days_start = thirty_days_ago - timedelta(days=30)
        previous_30_days_bookings = list(db.bookings.find({
            "createdAt": {"$gte": previous_30_days_start, "$lt": thirty_days_ago},
            "status": {"$in": ["confirmed", "completed"]}
        }))
        previous_30_days_revenue = 0
        for booking in previous_30_days_bookings:
            try:
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    previous_30_days_revenue += service["price"]
            except Exception:
                pass

        revenue_growth = ((
                                      last_30_days_revenue - previous_30_days_revenue) / previous_30_days_revenue) * 100 if previous_30_days_revenue > 0 else 0

        # Get package growth (compare active packages with previous month)
        current_active_packages = db.subscriptions.count_documents({
            "status": "active",
            "startDate": {"$gte": thirty_days_ago}
        })
        previous_active_packages = db.subscriptions.count_documents({
            "status": "active",
            "startDate": {"$gte": previous_30_days_start, "$lt": thirty_days_ago}
        })
        package_growth = ((
                                      current_active_packages - previous_active_packages) / previous_active_packages) * 100 if previous_active_packages > 0 else 0

        return {
            "totalUsers": total_users,
            "totalBookings": total_bookings,
            "totalRevenue": total_revenue,
            "activePackages": active_packages,
            "userGrowth": round(user_growth, 1),
            "bookingGrowth": round(booking_growth, 1),
            "revenueGrowth": round(revenue_growth, 1),
            "packageGrowth": round(package_growth, 1)
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get admin stats",
        )




@app.get("/analytics/chart-data", response_model=ChartDataResponse)
async def get_chart_data(timeframe: Timeframe = Timeframe.weekly):
    try:
        if timeframe == Timeframe.weekly:
            # Get data for the last 7 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)

            # Generate day names for the past 7 days
            days = [(start_date + timedelta(days=i)).strftime("%a") for i in range(7)]

            booking_data = []
            revenue_data = []

            for i, day in enumerate(days):
                day_start = start_date + timedelta(days=i)
                day_end = day_start + timedelta(days=1)

                # Count bookings for the day
                total_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": day_start, "$lt": day_end}
                })

                # Count completed bookings for the day
                completed_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": day_start, "$lt": day_end},
                    "status": "completed"
                })

                # Calculate revenue for the day
                day_bookings = db.bookings.find({
                    "bookingDate": {"$gte": day_start, "$lt": day_end},
                    "status": {"$in": ["confirmed", "completed"]}
                })

                day_revenue = 0
                for booking in day_bookings:
                    service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                    if service:
                        day_revenue += service["price"]

                booking_data.append({
                    "name": day,
                    "bookings": total_bookings,
                    "completed": completed_bookings
                })

                revenue_data.append({
                    "name": day,
                    "revenue": day_revenue
                })

        else:  # Monthly timeframe
            # Get data for the last 4 weeks
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(weeks=4)

            booking_data = []
            revenue_data = []

            for week in range(4):
                week_start = start_date + timedelta(weeks=week)
                week_end = week_start + timedelta(weeks=1)

                # Count bookings for the week
                total_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": week_start, "$lt": week_end}
                })

                # Count completed bookings for the week
                completed_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": week_start, "$lt": week_end},
                    "status": "completed"
                })

                # Calculate revenue for the week
                week_bookings = db.bookings.find({
                    "bookingDate": {"$gte": week_start, "$lt": week_end},
                    "status": {"$in": ["confirmed", "completed"]}
                })

                week_revenue = 0
                for booking in week_bookings:
                    service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                    if service:
                        week_revenue += service["price"]

                booking_data.append({
                    "name": f"Week {week + 1}",
                    "bookings": total_bookings,
                    "completed": completed_bookings
                })

                revenue_data.append({
                    "name": f"Week {week + 1}",
                    "revenue": week_revenue
                })

        return {
            "bookingData": booking_data,
            "revenueData": revenue_data
        }

    except Exception as e:
        logger.error(f"Error fetching chart data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chart data"
        )


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
