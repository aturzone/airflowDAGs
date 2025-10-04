"""
Alert Management System
=======================
Send notifications for critical anomalies via Slack, Email, etc.
"""

import logging
import requests
import json
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage anomaly alerts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def send_alerts(self, anomalies_df: pd.DataFrame):
        """Send alerts for anomalies"""
        
        if not self.enabled:
            logger.info("ℹ️ Alerts disabled")
            return
        
        # Group by severity
        critical = anomalies_df[anomalies_df['anomaly_score'] > 0.9]
        high = anomalies_df[(anomalies_df['anomaly_score'] > 0.8) & 
                           (anomalies_df['anomaly_score'] <= 0.9)]
        
        if len(critical) > 0:
            self._send_critical_alert(critical)
        
        if len(high) > 0:
            self._send_high_alert(high)
    
    def _send_critical_alert(self, anomalies: pd.DataFrame):
        """Send critical severity alert"""
        
        message = self._format_critical_message(anomalies)
        
        # Send via configured channels
        if 'slack' in self.config:
            self._send_slack(message, severity='critical')
        
        if 'email' in self.config:
            self._send_email(message, severity='critical')
        
        logger.info(f"🚨 Sent critical alert for {len(anomalies)} anomalies")
    
    def _send_high_alert(self, anomalies: pd.DataFrame):
        """Send high severity alert"""
        
        message = self._format_high_message(anomalies)
        
        if 'slack' in self.config:
            self._send_slack(message, severity='high')
        
        logger.info(f"⚠️ Sent high alert for {len(anomalies)} anomalies")
    
    def _format_critical_message(self, anomalies: pd.DataFrame) -> str:
        """Format critical alert message"""
        
        top_anomalies = anomalies.nlargest(5, 'anomaly_score')
        
        message = f"""
🚨 **CRITICAL ANOMALIES DETECTED** 🚨

Total Critical Anomalies: {len(anomalies)}

Top 5 Anomalies:
"""
        
        for idx, row in top_anomalies.iterrows():
            message += f"""
- Score: {row['anomaly_score']:.3f}
  Type: {row.get('anomaly_type', 'unknown')}
  Source: {row.get('source_table', 'unknown')}
"""
        
        message += f"""
⏰ Detected at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
🔗 View full report in Airflow UI
"""
        
        return message
    
    def _format_high_message(self, anomalies: pd.DataFrame) -> str:
        """Format high severity message"""
        
        message = f"""
⚠️ **High Severity Anomalies**

Total: {len(anomalies)}
Average Score: {anomalies['anomaly_score'].mean():.3f}

Types:
"""
        
        type_counts = anomalies['anomaly_type'].value_counts()
        for anom_type, count in type_counts.items():
            message += f"- {anom_type}: {count}\n"
        
        return message
    
    def _send_slack(self, message: str, severity: str = 'high'):
        """Send Slack notification"""
        
        slack_config = self.config.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("⚠️ Slack webhook URL not configured")
            return
        
        # Color coding
        colors = {
            'critical': '#FF0000',
            'high': '#FFA500',
            'medium': '#FFFF00'
        }
        
        payload = {
            'attachments': [{
                'color': colors.get(severity, '#CCCCCC'),
                'text': message,
                'mrkdwn_in': ['text']
            }]
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info("✅ Slack alert sent")
        except Exception as e:
            logger.error(f"❌ Slack alert failed: {e}")
    
    def _send_email(self, message: str, severity: str = 'high'):
        """Send email notification"""
        
        email_config = self.config.get('email', {})
        
        # Email sending would go here
        # Using SMTP or email service API
        
        logger.info(f"📧 Email alert would be sent (not implemented)")
    
    def _send_pagerduty(self, message: str, severity: str = 'high'):
        """Send PagerDuty alert"""
        
        pd_config = self.config.get('pagerduty', {})
        
        # PagerDuty API integration would go here
        
        logger.info(f"📟 PagerDuty alert would be sent (not implemented)")


class AlertingRules:
    """Define alerting rules and thresholds"""
    
    def __init__(self):
        self.rules = {
            'critical': {
                'anomaly_score_threshold': 0.9,
                'min_anomalies': 1,
                'notify': ['slack', 'email', 'pagerduty']
            },
            'high': {
                'anomaly_score_threshold': 0.8,
                'min_anomalies': 5,
                'notify': ['slack']
            },
            'medium': {
                'anomaly_score_threshold': 0.7,
                'min_anomalies': 10,
                'notify': []
            }
        }
    
    def evaluate(self, anomalies_df: pd.DataFrame) -> Dict:
        """Evaluate which rules are triggered"""
        
        triggered = {}
        
        for severity, rule in self.rules.items():
            matching = anomalies_df[
                anomalies_df['anomaly_score'] >= rule['anomaly_score_threshold']
            ]
            
            if len(matching) >= rule['min_anomalies']:
                triggered[severity] = {
                    'count': len(matching),
                    'notify_channels': rule['notify'],
                    'anomalies': matching
                }
        
        return triggered