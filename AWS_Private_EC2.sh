#!/bin/bash

# --- CONFIGURATION ---
INSTANCE_ID="i-04bc88b4faa2414f5"
# ---------------------

echo "🔑 Checking AWS SSO session validity..."
if aws sts get-caller-identity >/dev/null 2>&1; then
   echo "Valid SSO session"
else
   echo "Session expired, logging in..."
   aws sso login
fi

echo "🚀 Starting EC2 Instance: $INSTANCE_ID..."
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --query "StartingInstances[*].CurrentState.Name" --output text

echo "⏳ Waiting for EC2 instance state to change to 'running'..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
echo "✅ EC2 hardware state is 'running'."

echo "📡 Waiting for SSM Agent to ping 'Online'..."
echo "Note: This can take 1–3 minutes depending on your OS boot time."

while true; do
    # Fetch current SSM ping status
    SSM_STATUS=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query "InstanceInformationList[0].PingStatus" \
        --output text 2>/dev/null)

    # Clean up empty/null outputs
    if [ "$SSM_STATUS" == "None" ] || [ -z "$SSM_STATUS" ]; then
        SSM_STATUS="Initializing/Not_Managed"
    fi

    echo "👉 Current SSM Status: $SSM_STATUS"

    # Break loop only when status is strictly Online
    if [ "$SSM_STATUS" == "Online" ]; then
        echo "🎉 SSM Agent is Online!"
        break
    fi

    # Wait 10 seconds before polling again to respect AWS rate limits
    sleep 10
done

echo "🔌 Connecting to $INSTANCE_ID via SSM Session Manager..."
killall session-manager-plugin

# ssh port
sleep 2
aws ssm start-session --target "$INSTANCE_ID" --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["22"], "localPortNumber":["8443"]}' --region eu-west-1 &
# dcv port
sleep 2
aws ssm start-session --target "$INSTANCE_ID" --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["8443"], "localPortNumber":["8444"]}' --region eu-west-1 &
# # carla rpc port
# sleep 2
# aws ssm start-session --target "$INSTANCE_ID" --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["2000"], "localPortNumber":["2000"]}' --region eu-west-1 &
# # carla streaming port
# sleep 2
# aws ssm start-session --target "$INSTANCE_ID" --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["2001"], "localPortNumber":["2001"]}' --region eu-west-1 &

