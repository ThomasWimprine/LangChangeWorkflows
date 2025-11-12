# LoadBalancer Service Usage Guide

This guide explains how to create Kubernetes LoadBalancer services that will automatically receive external IP addresses and be advertised via BGP to your network infrastructure.

## Quick Start

To expose a service via LoadBalancer with automatic BGP advertisement:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: my-namespace
  labels:
    io.cilium/lb-pool: default      # Required: Select IP pool
    bgp: "true"                      # Required: Enable BGP advertisement
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
```

## Requirements

### 1. Service Labels (REQUIRED)

Your LoadBalancer service **MUST** include both labels:

```yaml
labels:
  io.cilium/lb-pool: <pool-name>   # Selects which IP pool to use
  bgp: "true"                       # Enables BGP advertisement
```

**Without these labels:**
- ❌ Service will remain in `<pending>` state (no IP assigned)
- ❌ Even if IP is assigned, it won't be routable (no BGP advertisement)

### 2. Available IP Pools

Choose one of the following pools based on your environment:

| Pool Name | Subnet | IP Range | Available IPs | Use Case |
|-----------|--------|----------|---------------|----------|
| `default-pool` | 192.168.100.0/24 | 192.168.100.10 - 192.168.100.254 | 245 | General purpose services |
| `dev-pool` | 192.168.101.0/24 | 192.168.101.10 - 192.168.101.254 | 245 | Development/testing |
| `test-pool` | 192.168.102.0/24 | 192.168.102.10 - 192.168.102.254 | 245 | Test environments |
| `prod-pool` | 192.168.103.0/24 | 192.168.103.10 - 192.168.103.254 | 245 | Production workloads |
| `mgmt-pool` | 192.168.200.0/24 | 192.168.200.10 - 192.168.200.254 | 245 | Management services |

**Note:** Each pool has its own dedicated /24 subnet for network isolation and easier firewall rule management.

### 3. Security Requirements for Talos

If your pods need to run on this Talos-based cluster, they **MUST** comply with security requirements:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      # Schedule on worker nodes (avoid control plane)
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-role.kubernetes.io/control-plane
                operator: DoesNotExist

      # Pod-level security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000              # Use non-root UID
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: my-app:latest

        # Container-level security context
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL

        # Use non-privileged ports (>1024)
        ports:
        - containerPort: 8080        # NOT 80

        # Required: Writable volumes for temp files
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /var/cache/app

      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
```

**Key Security Rules:**
- ✅ Run as non-root user (UID > 0)
- ✅ Read-only root filesystem
- ✅ Drop ALL capabilities
- ✅ No privilege escalation
- ✅ Use ports > 1024 (no privileged ports)
- ✅ Mount emptyDir volumes for writable paths

## Complete Example: nginx Service

```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-server
  template:
    metadata:
      labels:
        app: web-server
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-role.kubernetes.io/control-plane
                operator: DoesNotExist

      securityContext:
        runAsNonRoot: true
        runAsUser: 101
        fsGroup: 101
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: nginx
        image: nginx:1.25-alpine
        ports:
        - containerPort: 8080
          name: http

        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL

        volumeMounts:
        - name: cache
          mountPath: /var/cache/nginx
        - name: run
          mountPath: /var/run
        - name: nginx-conf
          mountPath: /etc/nginx/nginx.conf
          subPath: nginx.conf
          readOnly: true

        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi

      volumes:
      - name: cache
        emptyDir: {}
      - name: run
        emptyDir: {}
      - name: nginx-conf
        configMap:
          name: nginx-config

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
  namespace: production
data:
  nginx.conf: |
    user nginx;
    worker_processes auto;
    pid /var/run/nginx.pid;

    events {
        worker_connections 1024;
    }

    http {
        include /etc/nginx/mime.types;
        default_type application/octet-stream;

        access_log /dev/stdout;
        error_log /dev/stderr;

        server {
            listen 8080;  # Non-privileged port

            location / {
                root /usr/share/nginx/html;
                index index.html;
            }
        }
    }

---
apiVersion: v1
kind: Service
metadata:
  name: web-server
  namespace: production
  labels:
    io.cilium/lb-pool: prod-pool    # Required
    bgp: "true"                      # Required
spec:
  type: LoadBalancer
  selector:
    app: web-server
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
```

## Network Architecture

### BGP Advertisement

When you create a LoadBalancer service with the required labels:

1. **Cilium LB IPAM** assigns an IP from the selected pool
2. **Cilium BGP** advertises the IP to pfSense router (AS 65001)
3. **pfSense** adds route to routing table
4. **External clients** can reach the service via the LoadBalancer IP

```
┌─────────────────┐
│  External       │
│  Clients        │
└────────┬────────┘
         │
         │ http://192.168.100.10
         │
┌────────▼────────┐
│  pfSense        │  BGP Peering (AS 65001 ↔ AS 64513)
│  Router         │  Receives: 192.168.100.10/32
└────────┬────────┘  NextHop: 192.168.1.23 (lead control plane)
         │
         │ Routes to cluster
         │
┌────────▼────────────────────────────────┐
│  Kubernetes Cluster (AS 64513)          │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Cilium BGP (on control planes)   │  │
│  │ Advertises LoadBalancer IPs      │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Service: 192.168.100.10          │  │
│  │ Endpoints: 3 nginx pods          │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Load Balancing

Cilium provides eBPF-based load balancing:

- **Layer 4 (TCP/UDP)**: Socket-level load balancing
- **Algorithm**: Maglev consistent hashing
- **Mode**: Direct Server Return (DSR) for optimal performance
- **Health Checks**: Automatic endpoint health tracking

## Troubleshooting

### Service Stuck in `<pending>` State

**Check 1: Labels Present?**
```bash
kubectl get svc my-service -o yaml | grep -A2 labels
```

Expected output:
```yaml
labels:
  bgp: "true"
  io.cilium/lb-pool: default-pool
```

**Fix:** Add missing labels and re-apply service.

---

**Check 2: IP Pool Available?**
```bash
kubectl get ciliumloadbalancerippool
```

Expected output:
```
NAME           DISABLED   CONFLICTING   IPS AVAILABLE   AGE
default-pool   false      False         >0              12h
```

**Fix:** If `IPS AVAILABLE` is 0, choose a different pool or contact cluster admin.

---

### Service Has IP But Not Reachable

**Check 1: BGP Advertisement Enabled?**
```bash
kubectl get svc my-service -o jsonpath='{.metadata.labels.bgp}'
```

Expected output: `true`

**Fix:** Add label `bgp: "true"` to service.

---

**Check 2: BGP Route Advertised?**
```bash
kubectl exec -n cilium ds/cilium -- cilium bgp routes | grep <EXTERNAL-IP>
```

Expected output:
```
64513     192.168.100.10/32    0.0.0.0   Active
```

**Fix:** If missing, verify service has `bgp: "true"` label.

---

**Check 3: BGP Peering Established?**
```bash
kubectl exec -n cilium ds/cilium -- cilium bgp peers
```

Expected output:
```
Local AS   Peer AS   Peer Address       Session       Uptime
64513      65001     192.168.1.99:179   established   10h
```

**Fix:** If session is not established, contact cluster admin.

---

### Pods Not Scheduling

**Check: Pod Security Context?**
```bash
kubectl describe pod <pod-name> | grep -A10 "securityContext"
```

Required settings:
- `runAsNonRoot: true`
- `runAsUser: <non-zero>`
- `readOnlyRootFilesystem: true`
- `capabilities.drop: [ALL]`

**Fix:** Update pod spec with required security context (see example above).

## Reserved IPs

The following IPs are reserved and cannot be used by LoadBalancer services:

| IP | Purpose |
|----|---------|
| 192.168.100.103 | Kubernetes API VIP (control plane HA) |

## Contact

For issues or questions:
- **Cluster Admin**: [Your contact info]
- **Documentation**: This guide
- **Validation Script**: See section below

## Validation Script

Use this script to validate your service configuration before deployment:

```bash
#!/bin/bash
# validate-loadbalancer.sh

SERVICE_FILE="$1"

if [ -z "$SERVICE_FILE" ]; then
    echo "Usage: $0 <service.yaml>"
    exit 1
fi

echo "Validating LoadBalancer service configuration..."

# Check for required labels
HAS_POOL=$(grep "io.cilium/lb-pool:" "$SERVICE_FILE")
HAS_BGP=$(grep 'bgp: "true"' "$SERVICE_FILE")

if [ -z "$HAS_POOL" ]; then
    echo "❌ FAIL: Missing required label 'io.cilium/lb-pool'"
    exit 1
fi

if [ -z "$HAS_BGP" ]; then
    echo "❌ FAIL: Missing required label 'bgp: \"true\"'"
    exit 1
fi

# Check service type
HAS_LB_TYPE=$(grep "type: LoadBalancer" "$SERVICE_FILE")

if [ -z "$HAS_LB_TYPE" ]; then
    echo "❌ FAIL: Service type must be 'LoadBalancer'"
    exit 1
fi

echo "✅ PASS: Service has required labels"
echo "✅ PASS: Service type is LoadBalancer"
echo ""
echo "Configuration looks good! Deploy with:"
echo "  kubectl apply -f $SERVICE_FILE"
```

Save as `validate-loadbalancer.sh`, make executable, and run:
```bash
chmod +x validate-loadbalancer.sh
./validate-loadbalancer.sh my-service.yaml
```
