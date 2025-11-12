# Kubernetes Cluster Reference Configurations

This document provides the essential configuration files and patterns used in this Talos/Cilium/BGP Kubernetes cluster. Copy these configurations to your project repositories as starting templates.

## Table of Contents

1. [IP Pool Configuration](#ip-pool-configuration)
2. [BGP Configuration](#bgp-configuration)
3. [LoadBalancer Service Template](#loadbalancer-service-template)
4. [Deployment Security Template](#deployment-security-template)
5. [Network Architecture](#network-architecture)

---

## IP Pool Configuration

**File:** `kubernetes/cilium/vip-pools.yaml`

This defines the five LoadBalancer IP pools, each with its own /24 subnet:

```yaml
---
# Cilium LoadBalancer IP Pools for Environment-Based VIP Management
#
# Pool Assignment Strategy:
# - 192.168.100.0/24: Default/Infrastructure (cluster services, ingress, monitoring)
# - 192.168.101.0/24: Development environment
# - 192.168.102.0/24: Test/Staging environment
# - 192.168.103.0/24: Production environment
# - 192.168.200.0/24: Management/Infrastructure (Hubble, Prometheus, Grafana)

---
# Pool 1: Default/Infrastructure (192.168.100.0/24)
apiVersion: cilium.io/v2alpha1
kind: CiliumLoadBalancerIPPool
metadata:
  name: default-pool
spec:
  blocks:
    - start: "192.168.100.10"
      stop: "192.168.100.254"
  serviceSelector:
    matchLabels:
      io.cilium/lb-pool: default
  disabled: false

---
# Pool 2: Development Environment (192.168.101.0/24)
apiVersion: cilium.io/v2alpha1
kind: CiliumLoadBalancerIPPool
metadata:
  name: dev-pool
spec:
  blocks:
    - start: "192.168.101.10"
      stop: "192.168.101.254"
  serviceSelector:
    matchLabels:
      io.cilium/lb-pool: dev
  disabled: false

---
# Pool 3: Test/Staging Environment (192.168.102.0/24)
apiVersion: cilium.io/v2alpha1
kind: CiliumLoadBalancerIPPool
metadata:
  name: test-pool
spec:
  blocks:
    - start: "192.168.102.10"
      stop: "192.168.102.254"
  serviceSelector:
    matchLabels:
      io.cilium/lb-pool: test
  disabled: false

---
# Pool 4: Production Environment (192.168.103.0/24)
apiVersion: cilium.io/v2alpha1
kind: CiliumLoadBalancerIPPool
metadata:
  name: prod-pool
spec:
  blocks:
    - start: "192.168.103.10"
      stop: "192.168.103.254"
  serviceSelector:
    matchLabels:
      io.cilium/lb-pool: prod
  disabled: false

---
# Pool 5: Management/Infrastructure (192.168.200.0/24)
apiVersion: cilium.io/v2alpha1
kind: CiliumLoadBalancerIPPool
metadata:
  name: mgmt-pool
spec:
  blocks:
    - start: "192.168.200.10"
      stop: "192.168.200.254"
  serviceSelector:
    matchLabels:
      io.cilium/lb-pool: mgmt
  disabled: false
```

### Reserved IPs

| IP | Purpose |
|----|---------|
| 192.168.100.103 | Kubernetes API VIP (control plane HA) |
| 192.168.100.1-9 | Reserved for infrastructure |
| 192.168.101.1-9 | Reserved for infrastructure |
| 192.168.102.1-9 | Reserved for infrastructure |
| 192.168.103.1-9 | Reserved for infrastructure |
| 192.168.200.1-9 | Reserved for infrastructure |

---

## BGP Configuration

### BGP Peering Configuration

**File:** `kubernetes/cilium/bgp-peer-config.yaml`

```yaml
---
apiVersion: cilium.io/v2alpha1
kind: CiliumBGPPeeringPolicy
metadata:
  name: bgp-peering-policy
spec:
  nodeSelector:
    matchLabels:
      node-role.kubernetes.io/control-plane: ""
  virtualRouters:
  - localASN: 64513
    exportPodCIDR: false
    neighbors:
    - peerAddress: "192.168.1.99/32"
      peerASN: 65001
      eBGPMultihopTTL: 10
      connectRetryTimeSeconds: 120
      holdTimeSeconds: 90
      keepAliveTimeSeconds: 30
```

**Configuration Details:**
- **Local ASN**: 64513 (Kubernetes cluster)
- **Peer ASN**: 65001 (pfSense router)
- **Peer Address**: 192.168.1.99 (pfSense BGP daemon)
- **Node Selection**: Only control plane nodes participate in BGP
- **Timers**:
  - Connect retry: 120s
  - Hold time: 90s
  - Keep-alive: 30s

### BGP Advertisement Policy

**File:** `kubernetes/cilium/bgp-advertisement.yaml`

```yaml
---
# Advertise LoadBalancer service IPs that are tagged with bgp=true
apiVersion: cilium.io/v2
kind: CiliumBGPAdvertisement
metadata:
  name: lb-service-advertisement
  labels:
    advertise: bgp
spec:
  advertisements:
  - advertisementType: Service
    service:
      addresses:
      - LoadBalancerIP
    selector:
      matchExpressions:
      - key: bgp
        operator: In
        values:
        - "true"

---
# Advertise the Kubernetes API VIP
apiVersion: cilium.io/v2
kind: CiliumBGPAdvertisement
metadata:
  name: lb-vip-advertisement
  labels:
    advertise: bgp
spec:
  advertisements:
  - advertisementType: Service
    service:
      addresses:
      - LoadBalancerIP
    selector:
      matchLabels:
        io.kubernetes.service.name: kube-apiserver-lb
```

**Key Points:**
- Services need label `bgp: "true"` to be advertised
- LoadBalancer IPs are advertised as /32 host routes
- Kubernetes API VIP is automatically advertised

---

## LoadBalancer Service Template

### Basic Template

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

### Environment-Specific Examples

**Development Service:**
```yaml
metadata:
  name: dev-api
  namespace: development
  labels:
    io.cilium/lb-pool: dev          # Uses 192.168.101.0/24
    bgp: "true"
```

**Production Service:**
```yaml
metadata:
  name: prod-api
  namespace: production
  labels:
    io.cilium/lb-pool: prod         # Uses 192.168.103.0/24
    bgp: "true"
```

**Management/Monitoring Service:**
```yaml
metadata:
  name: prometheus
  namespace: monitoring
  labels:
    io.cilium/lb-pool: mgmt         # Uses 192.168.200.0/24
    bgp: "true"
```

---

## Deployment Security Template

All deployments on this Talos cluster **MUST** use the following security context:

### Complete Secure Deployment Template

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: my-namespace
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      # Schedule on worker nodes only (avoid control plane)
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
        runAsUser: 1000              # Use non-root UID (app-specific)
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: my-app:latest

        # Container-level security context (REQUIRED)
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL

        # Use non-privileged ports (>1024)
        ports:
        - containerPort: 8080        # NOT 80

        # Mount writable volumes for temp/cache
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /var/cache/app

        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi

      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
```

### nginx Example with Custom Config

```yaml
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
        runAsUser: 101               # nginx user UID
        fsGroup: 101
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: nginx
        image: nginx:1.25-alpine

        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL

        ports:
        - containerPort: 8080        # Non-privileged port
          name: http

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

            location /health {
                return 200 "healthy\n";
                add_header Content-Type text/plain;
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
    io.cilium/lb-pool: prod         # Production pool
    bgp: "true"                      # Enable BGP
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

### Security Checklist

- ✅ `runAsNonRoot: true` (pod and container level)
- ✅ `runAsUser: <non-zero>` (use app-specific UID)
- ✅ `readOnlyRootFilesystem: true`
- ✅ `capabilities.drop: [ALL]`
- ✅ `allowPrivilegeEscalation: false`
- ✅ `seccompProfile.type: RuntimeDefault`
- ✅ Use ports > 1024 (no privileged ports)
- ✅ Mount emptyDir for `/tmp`, `/var/cache`, `/var/run`
- ✅ Define resource requests and limits
- ✅ Use node affinity to avoid control plane

---

## Network Architecture

### Cluster Overview

```
┌─────────────────────────────────────────────────────────────┐
│  External Network (192.168.1.0/24)                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  pfSense Router (192.168.1.99)                       │  │
│  │  BGP AS: 65001                                       │  │
│  │  - Receives /32 routes from cluster                  │  │
│  │  - Routes traffic to Kubernetes services             │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │ BGP Peering                             │
│                   │ AS 65001 ↔ AS 64513                     │
└───────────────────┼─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│  Kubernetes Cluster (192.168.1.0/24 physical)               │
│  BGP AS: 64513                                              │
│                                                              │
│  Control Plane Nodes (BGP enabled):                         │
│  ├─ lead   (192.168.1.23)   - PRIMARY BGP speaker          │
│  ├─ nickel (192.168.1.21)   - BACKUP BGP speaker           │
│  └─ tin    (192.168.1.25)   - BACKUP BGP speaker           │
│                                                              │
│  Worker Nodes (no BGP):                                     │
│  ├─ barrel (192.168.1.47)   - 7.5Gi RAM                     │
│  └─ cobalt (192.168.1.158)  - 15.5Gi RAM                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Cilium CNI with BGP Control Plane                 │    │
│  │  - Manages LoadBalancer IP allocation              │    │
│  │  - Advertises IPs via BGP                          │    │
│  │  - eBPF-based load balancing (no kube-proxy)       │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  LoadBalancer IP Pools (BGP advertised):                    │
│  ├─ default-pool: 192.168.100.0/24 (245 IPs)               │
│  ├─ dev-pool:     192.168.101.0/24 (245 IPs)               │
│  ├─ test-pool:    192.168.102.0/24 (245 IPs)               │
│  ├─ prod-pool:    192.168.103.0/24 (245 IPs)               │
│  └─ mgmt-pool:    192.168.200.0/24 (245 IPs)               │
│                                                              │
│  Reserved IPs:                                              │
│  └─ 192.168.100.103 - Kubernetes API VIP                    │
└──────────────────────────────────────────────────────────────┘
```

### Traffic Flow Example

1. **User requests** `http://192.168.100.10` (LoadBalancer IP)
2. **pfSense** receives request, checks routing table
3. **BGP route** shows `192.168.100.10/32` → NextHop: `192.168.1.23` (lead)
4. **pfSense** forwards to `192.168.1.23`
5. **Cilium eBPF** on lead performs socket-level load balancing
6. **Traffic distributed** to backend pods (on barrel/cobalt workers)
7. **Response** sent directly back to client (Direct Server Return)

### Key Features

- **BGP Control Plane**: Control plane nodes advertise LoadBalancer IPs
- **eBPF Load Balancing**: No kube-proxy, socket-level LB
- **Maglev Hashing**: Consistent connection tracking
- **Health Checking**: Automatic endpoint health tracking
- **Direct Server Return**: Optimized response path
- **Network Isolation**: Separate /24 per environment

---

## Quick Reference Commands

### Check BGP Status
```bash
# BGP peering status
kubectl exec -n cilium ds/cilium -- cilium bgp peers

# BGP routes being advertised
kubectl exec -n cilium ds/cilium -- cilium bgp routes
```

### Check IP Pool Allocation
```bash
# List all IP pools
kubectl get ciliumloadbalancerippool

# Check specific pool details
kubectl get ciliumloadbalancerippool default-pool -o yaml
```

### Check LoadBalancer Services
```bash
# List all LoadBalancer services with external IPs
kubectl get svc -A --field-selector spec.type=LoadBalancer

# Check specific service
kubectl get svc my-service -o wide
```

### Troubleshooting
```bash
# Cilium status
kubectl -n cilium exec ds/cilium -- cilium status

# Service endpoints
kubectl get endpoints my-service

# Check BGP advertisement
kubectl exec -n cilium ds/cilium -- cilium bgp routes | grep <EXTERNAL-IP>
```

---

## Common Patterns

### Pattern 1: Simple Web Application

```yaml
# Deployment + Service in one file
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
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
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: app
        image: my-web-app:v1.0
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: [ALL]
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: web-app
  namespace: production
  labels:
    io.cilium/lb-pool: prod
    bgp: "true"
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8080
```

### Pattern 2: Development vs Production

**Development:**
```yaml
labels:
  io.cilium/lb-pool: dev          # 192.168.101.x
  bgp: "true"
  environment: development
```

**Production:**
```yaml
labels:
  io.cilium/lb-pool: prod         # 192.168.103.x
  bgp: "true"
  environment: production
```

### Pattern 3: Internal-Only Service (No BGP)

```yaml
# Service without BGP advertisement (cluster-internal only)
metadata:
  name: internal-service
  namespace: default
  labels:
    io.cilium/lb-pool: default
    # Note: No bgp: "true" label
spec:
  type: LoadBalancer
  # Gets IP but NOT advertised via BGP
```

---

## Additional Documentation

- **Full LoadBalancer Guide**: See [LOADBALANCER-USAGE-GUIDE.md](LOADBALANCER-USAGE-GUIDE.md)
- **Cluster Configuration**: See [talos/](talos/) directory
- **Cilium Manifests**: See [kubernetes/cilium/](kubernetes/cilium/) directory
