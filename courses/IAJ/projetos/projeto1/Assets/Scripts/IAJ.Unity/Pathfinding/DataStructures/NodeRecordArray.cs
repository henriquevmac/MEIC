using Assets.Scripts.Grid;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Assets.Scripts.IAJ.Unity.Pathfinding.DataStructures
{
    public class NodeRecordArray : IOpenSet, IClosedSet
    {
        private NodeRecord[] NodeRecords { get; set; }
        private NodePriorityHeap Open { get; set; }

        public NodeRecordArray(List<Node> nodes)
        {
            //this method creates and initializes the NodeRecordArray for all nodes in the Navigation Graph
            this.NodeRecords = new NodeRecord[nodes.Count];
            
            for(int i = 0; i < nodes.Count; i++)
            {
                nodes[i].index = i;
                this.NodeRecords[i] = new NodeRecord(nodes[i]);
            }

            this.Open = new NodePriorityHeap();
        }

        public NodeRecord GetNodeRecord(NodeRecord node)
        {
            return NodeRecords[node.Node.index];
        }

        private NodeRecord UpdateNodeRecord(NodeRecord nodeRecord)
        {
            NodeRecord storedNodeRecord = GetNodeRecord(nodeRecord);
            storedNodeRecord.parent = nodeRecord.parent;
            storedNodeRecord.gCost = nodeRecord.gCost;
            storedNodeRecord.hCost = nodeRecord.hCost;
            storedNodeRecord.fCost = nodeRecord.fCost;
            return storedNodeRecord;
        }

        void IOpenSet.Clear()
        {
            this.Open.Clear();
            //we want this to be very efficient (that's why we use for)
            for (int i = 0; i < this.NodeRecords.Length; i++)
            {
                if(NodeRecords[i].Node.isWalkable)
                this.NodeRecords[i].status = NodeStatus.Unvisited;
            }

        }

        void IClosedSet.Clear()
        {
        
        }

        
        void IOpenSet.Add(NodeRecord nodeRecord)
        {
            // TODO implement
            throw new NotImplementedException();
        }

        void IClosedSet.Add(NodeRecord nodeRecord)
        {
            // TODO implement
            throw new NotImplementedException();
        }

        NodeRecord IOpenSet.Find(NodeRecord nodeRecord)
        {
            //TODO implement
            throw new NotImplementedException();
        }

        NodeRecord IClosedSet.Find(NodeRecord nodeRecord)
        {
            //TODO implement
            throw new NotImplementedException();
        }

        public NodeRecord GetBestAndRemove()
        {
            return this.Open.GetBestAndRemove();
        }

        public NodeRecord PeekBest()
        {
            return this.Open.PeekBest();
        }

        public void Replace(NodeRecord nodeToBeReplaced, NodeRecord nodeToReplace)
        {
            NodeRecord storedNodeRecord = UpdateNodeRecord(nodeToReplace);
            this.Open.Replace(nodeToBeReplaced, storedNodeRecord);
        }

        void IOpenSet.Remove(NodeRecord nodeRecord)
        {
            //TODO implement
            throw new NotImplementedException();
        }

        void IClosedSet.Remove(NodeRecord nodeRecord)
        {
            //TODO implement
            throw new NotImplementedException();
        }

        ICollection<NodeRecord> IOpenSet.All()
        {
            return this.Open.All();
        }

        ICollection<NodeRecord> IClosedSet.All()
        {
            return this.NodeRecords.Where(node => node.status == NodeStatus.Closed).ToList();
        }

        public int CountOpen()
        {
            return this.Open.CountOpen();
        }
    }
}
