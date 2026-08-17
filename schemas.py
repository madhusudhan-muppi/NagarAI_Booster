from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Complaint(BaseModel):
    id: str
    citizen_id: str
    raw_text: str
    transcript: Optional[str] = None
    photo_url: Optional[str] = None
    category: str  # "pothole", "garbage", "streetlight", "waterlogging", "other"
    severity: int  # 1–5
    location_lat: float
    location_lon: float
    location_text: Optional[str] = None
    description_en: str
    created_at: datetime
    days_pending: int

class ClusterSummary(BaseModel):
    cluster_id: str
    category: str
    location_lat: float
    location_lon: float
    affected_citizens: int
    severity_max: int
    days_pending_max: int
    priority_score: float
    status: str  # "open", "in_progress", "resolved"
    description: str
    member_complaint_ids: List[str]

class PriorityBreakdown(BaseModel):
    complaint_id: str
    severity_component: float
    headcount_component: float
    ageing_component: float
    proximity_component: float
    hazard_lane: bool
    final_score: float
    formula_used: str
