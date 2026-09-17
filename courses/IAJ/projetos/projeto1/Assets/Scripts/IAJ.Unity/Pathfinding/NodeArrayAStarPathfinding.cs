using Assets.Scripts.IAJ.Unity.Pathfinding.Heuristics;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Grid;
using Assets.Scripts.IAJ.Unity.Pathfinding.DataStructures;
using System.Runtime.CompilerServices;

namespace Assets.Scripts.IAJ.Unity.Pathfinding
{
    public class NodeArrayAStarPathfinding : AStarPathfinding
    {
        private static int index = 0;
        protected NodeRecordArray NodeRecordArray { get; set; }

        public NodeArrayAStarPathfinding(IGraph grid,IHeuristic heuristic, float tieBreakingWeight) : base(grid,null, null, heuristic, tieBreakingWeight)
        {
     
            this.NodeRecordArray = new NodeRecordArray(grid.GetAll());
            this.Open = this.NodeRecordArray;
            this.Closed = this.NodeRecordArray;
            this.NodesPerFrame = 20;

        }

        // In Node Array A* the only thing that changes is how you process the child node, the search occurs the exact same way so you can the parent's method       
        /*protected override void ProcessChildNode(NodeRecord parentNode, NodeRecord node)
        {
            // TODO implement
            
            // this.TotalProcessedNodes++
        }*/
    }
}
               

