# db_prep/runbooks/IOException_replication.md

## Root Cause: IOException during block replication

### Symptoms
- Log event E26: "Received exception while serving blk_ ... java.io.IOException"
- Replication pipeline interrupted before completing

### Diagnosis steps
1. Check which DataNode emitted the exception
2. Verify disk space on that node
3. Review network connectivity between nodes

### Remediation
- Restart affected DataNode
- Re-trigger replication via `hdfs fsck /`