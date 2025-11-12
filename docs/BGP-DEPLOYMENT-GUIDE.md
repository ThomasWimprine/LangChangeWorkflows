# Cilium BGP VIP Management Deployment Guide

## Overview

This guide walks through deploying BGP-based LoadBalancer VIP management for Cilium CNI. After deployment, Kubernetes LoadBalancer services will receive VIPs that are automatically advertised via BGP to the pfSense router.

**What You'll Deploy:**
- Cilium BGP Peering Policy (worker nodes ↔ pfSense)
- 4 LoadBalancer IP pools (control-plane, ingress, internal, monitoring)
- BGP authentication with MD5 passwords
- Automatic /32 route advertisement for VIPs

---

## Prerequisites

✅ **Completed:**
- Talos Linux cluster deployed (3 control planes + 6 workers)
- Cilium CNI installed (v1.16.5)
- All nodes in Ready state
- pfSense router at 192.168.100.1

✅ **Required Tools:**
- kubectl configured for cluster access
- SOPS installed for secret encryption
- Age key available: `~/.config/sops/age/keys.txt`

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     pfSense Router                          │
│                  AS 65001 (BGP Peer)                        │
│                   192.168.100.1                             │
└────────────┬────────────────────────────────────────────────┘
             │
             │ BGP Sessions (6 sessions)
             │
    ┌────────┴────────────────────────────────┐
    │                                         │
┌───┴────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ apple  │ │blueberry│ │ cherry │ │ lemon  │ │ pecan  │ │rubharb │
│  .201  │ │  .202   │ │  .203  │ │  .204  │ │  .205  │ │  .206  │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    AS 64513 Workers (Cilium BGP)

    Each worker advertises /32 VIP routes
    ↓
    VIP Pools:
    - 192.168.100.0/24 (control plane)
    - 192.168.101.0/24 (ingress/public)
    - 192.168.102.0/24 (internal)
    - 192.168.103.0/24 (monitoring)
```

---

## Deployment Steps

### Step 1: Configure BGP Authentication Secret

First, create a strong BGP MD5 password:

```bash
# Generate a strong password
BGP_PASSWORD=$(openssl rand -base64 32)
echo "Generated BGP Password: $BGP_PASSWORD"

# Save to temporary file for SOPS encryption
cat > /tmp/bgp-secret.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: cilium-bgp-auth
  namespace: kube-system
type: Opaque
stringData:
  password: "$BGP_PASSWORD"
EOF

# Encrypt with SOPS
cd /home/thomas/Repositories/AWS_Environment
sops --encrypt /tmp/bgp-secret.yaml > kubernetes/cilium/bgp-secret.enc.yaml

# Clean up temporary file
rm /tmp/bgp-secret.yaml

# Verify encryption
sops --decrypt kubernetes/cilium/bgp-secret.enc.yaml
```

**Important:** Save the `BGP_PASSWORD` value - you'll need it for pfSense configuration.

---

### Step 2: Apply Encrypted Secret to Cluster

```bash
# Decrypt and apply secret to cluster
sops --decrypt kubernetes/cilium/bgp-secret.enc.yaml | kubectl apply -f -

# Verify secret created
kubectl get secret -n kube-system cilium-bgp-auth

# Expected output:
# NAME               TYPE     DATA   AGE
# cilium-bgp-auth    Opaque   1      5s
```

---

### Step 3: Apply BGP Configuration

```bash
# Apply BGP peering policy
kubectl apply -f kubernetes/cilium/bgp-config.yaml

# Verify BGP policy created
kubectl get ciliumbgppeeringpolicy -n kube-system

# Expected output:
# NAME                    AGE
# pfsense-bgp-peering     10s
```

---

### Step 4: Apply VIP Pools

```bash
# Apply LoadBalancer IP pools
kubectl apply -f kubernetes/cilium/vip-pools.yaml

# Verify pools created
kubectl get ciliumloadbalancerippool -n kube-system

# Expected output:
# NAME          AGE
# default-pool  5s
# dev-pool      5s
# mgmt-pool     5s
# prod-pool     5s
# test-pool     5s
```

---

### Step 5: Configure pfSense BGP

Follow the detailed guide: [`docs/PFSENSE-BGP-CONFIGURATION.md`](PFSENSE-BGP-CONFIGURATION.md)

**Quick Summary:**
1. Install FRR package on pfSense
2. Enable BGP with AS 65001
3. Add 6 BGP neighbors (one per worker: 192.168.100.201-206)
4. Configure MD5 authentication with same password from Step 1
5. Set timers: keepalive 3s, hold 9s
6. Apply route-map to accept VIP pools
7. Save and apply configuration

---

### Step 6: Validate BGP Sessions

```bash
# Run automated validation tests
cd /home/thomas/Repositories/AWS_Environment
./tests/infrastructure/bgp-validation.sh --verbose

# Expected output:
# ✅ All validations PASSED
# BGP Sessions: 6 established
```

**From pfSense CLI:**
```bash
vtysh -c "show bgp summary"

# Expected: 6 neighbors in "Established" state
```

---

## Testing LoadBalancer VIPs

### Test 1: Ingress Pool Service

Create a test service in the ingress pool:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: test-nginx-ingress
  namespace: default
  labels:
    io.cilium/lb-pool: ingress
    advertise: bgp
spec:
  type: LoadBalancer
  loadBalancerIP: 192.168.101.10
  ports:
    - port: 80
      targetPort: 80
      protocol: TCP
  selector:
    app: nginx-test
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-test
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx-test
  template:
    metadata:
      labels:
        app: nginx-test
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
EOF
```

**Verify:**
```bash
# Check service VIP assigned
kubectl get svc test-nginx-ingress

# Expected output:
# NAME                 TYPE           CLUSTER-IP      EXTERNAL-IP       PORT(S)
# test-nginx-ingress   LoadBalancer   100.68.x.x      192.168.101.10    80:xxxxx/TCP

# Check BGP route advertisement from pfSense
vtysh -c "show bgp ipv4 unicast 192.168.101.10/32"

# Test connectivity
curl http://192.168.101.10
# Expected: Nginx welcome page
```

---

### Test 2: Monitoring Pool Service

Create a Prometheus service in management pool:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
  labels:
    io.cilium/lb-pool: mgmt
    advertise: bgp
spec:
  type: LoadBalancer
  loadBalancerIP: 192.168.200.11
  ports:
    - port: 9090
      targetPort: 9090
      protocol: TCP
  selector:
    app: prometheus
EOF
```

**Note:** This assumes Prometheus is already deployed. Adjust selector as needed.

---

## Monitoring and Troubleshooting

### Check BGP Session Status

```bash
# From Kubernetes cluster
kubectl exec -n kube-system <cilium-pod> -- cilium bgp peers

# From pfSense
vtysh -c "show bgp summary"
```

### View Advertised Routes

```bash
# From Cilium pod
kubectl exec -n kube-system <cilium-pod> -- cilium bgp routes

# From pfSense
vtysh -c "show bgp ipv4 unicast"
```

### Debug BGP Issues

```bash
# Check Cilium BGP peering policy
kubectl describe ciliumbgppeeringpolicy -n kube-system pfsense-bgp-peering

# Check Cilium agent logs
kubectl logs -n kube-system <cilium-pod> | grep -i bgp

# Check pfSense BGP logs
tail -f /var/log/frr/bgpd.log
```

---

## Common Issues

### Issue 1: BGP Sessions Not Establishing

**Symptoms:**
- BGP sessions show "Active" or "Connect" state
- No routes received

**Solutions:**
1. Verify MD5 password matches on both sides
2. Check network connectivity between workers and pfSense
3. Verify firewall rules allow TCP port 179
4. Check Cilium pod logs for errors

### Issue 2: VIP Not Assigned to Service

**Symptoms:**
- Service shows `EXTERNAL-IP: <pending>`

**Solutions:**
1. Check service has correct pool label: `io.cilium/lb-pool: <pool-name>`
2. Verify IP pool exists: `kubectl get ciliumloadbalancerippool -n kube-system`
3. Check IP pool CIDR includes requested IP
4. Review Cilium operator logs

### Issue 3: VIP Not Reachable

**Symptoms:**
- VIP assigned but ping/curl fails

**Solutions:**
1. Verify BGP route in pfSense routing table: `vtysh -c "show ip route"`
2. Check route advertisement: `vtysh -c "show bgp ipv4 unicast <VIP>/32"`
3. Verify pods are running: `kubectl get pods -l app=<selector>`
4. Test from worker node directly

---

## Production Considerations

### 1. Expand VIP Pool Ranges

The initial configuration uses single /32 IPs for testing. For production:

```yaml
# Edit kubernetes/cilium/vip-pools.yaml
spec:
  blocks:
    - cidr: "192.168.101.0/24"  # Full /24 range instead of single /32
```

### 2. BGP Security Hardening

- Use strong MD5 passwords (32+ characters)
- Rotate passwords periodically
- Restrict BGP access with firewall rules
- Monitor BGP session state with alerts

### 3. Route Filtering

Configure strict prefix filters on pfSense to accept only expected VIP ranges.

### 4. High Availability

- BGP provides automatic failover if a worker fails
- Multiple workers can advertise the same VIP (anycast)
- Test failover by cordoning a node:
  ```bash
  kubectl cordon apple
  # Verify VIP still reachable
  ```

### 5. Monitoring and Alerting

Set up alerts for:
- BGP session down
- Missing route advertisements
- VIP unreachable
- BGP flapping (frequent session resets)

---

## Rollback Procedure

If issues occur, rollback in reverse order:

```bash
# 1. Delete VIP pools
kubectl delete -f kubernetes/cilium/vip-pools.yaml

# 2. Delete BGP peering policy
kubectl delete -f kubernetes/cilium/bgp-config.yaml

# 3. Delete BGP secret
kubectl delete secret -n kube-system cilium-bgp-auth

# 4. Disable BGP on pfSense
# Navigate to: Services → FRR → BGP
# Uncheck "Enable" and save
```

---

## Next Steps

After successful BGP deployment:

1. ✅ Deploy ingress controller with LoadBalancer VIP
2. ✅ Configure monitoring stack (Prometheus/Grafana) with VIPs
3. ✅ Set up Hubble UI with LoadBalancer access
4. ✅ Implement network policies for VIP access control
5. ✅ Configure automated monitoring and alerting

---

## References

- [Cilium BGP Documentation](https://docs.cilium.io/en/stable/network/bgp-control-plane/)
- [pfSense FRR BGP](https://docs.netgate.com/pfsense/en/latest/packages/frr/bgp.html)
- [BGP Best Practices](https://datatracker.ietf.org/doc/html/rfc7454)

---

## Support

For deployment assistance:
1. Check validation script output: `./tests/infrastructure/bgp-validation.sh --verbose`
2. Review Cilium documentation: https://docs.cilium.io
3. Check pfSense FRR logs: `/var/log/frr/bgpd.log`
4. Consult troubleshooting section above
