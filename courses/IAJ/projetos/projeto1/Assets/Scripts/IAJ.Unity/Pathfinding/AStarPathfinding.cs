using Assets.Scripts.IAJ.Unity.Pathfinding.Heuristics;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Grid;
using Assets.Scripts.IAJ.Unity.Pathfinding.DataStructures;
using System.Runtime.CompilerServices;
using System;
using UnityEditor.Experimental.GraphView;
using Node = Assets.Scripts.Grid.Node;
using UnityEngine.Assertions.Must;

namespace Assets.Scripts.IAJ.Unity.Pathfinding
{
    [Serializable]
    public class AStarPathfinding
    {
        protected float TieBreakingWeight;
        PathfindingManager pathfindingManager;
        public IGraph gridGraph { get; set; }
        public uint NodesPerFrame { get; set; }
        public uint TotalProcessedNodes { get; protected set; }
        public int MaxOpenNodes { get; protected set; }
        public float TotalProcessingTime { get; set; }
        public bool InProgress { get; set; }
        public IOpenSet Open { get; protected set; }
        public IClosedSet Closed { get; protected set; }
        public IHeuristic Heuristic { get; protected set; }

        public Node GoalNode { get; set; }
        public Node StartNode { get; set; }
        public int StartPositionX { get; set; }
        public int StartPositionY { get; set; }
        public int GoalPositionX { get; set; }
        public int GoalPositionY { get; set; }


        // variables for analysis purposes
        public int AStarPathfindingSearchCalls { get; set; } = 0;
        
        public int GetBestAndRemoveCalls { get; set; } = 0;
        
        public int AddToOpenCalls { get; set; } = 0;
        
        public int SearchInOpenCalls { get; set; } = 0;

        public int RemoveFromOpenCalls { get; set; } = 0;

        public int ReplaceCalls { get; set; } = 0;

        public int AddToClosedCalls { get; set; } = 0;

        public int SearchInClosedCalls { get; set; } = 0;

        public int RemoveFromClosedCalls { get; set; } = 0;


        public AStarPathfinding(IGraph grid, IOpenSet open, IClosedSet closed, IHeuristic heuristic, float tieBreakingWeight)
        {
            this.gridGraph = grid;  
            this.Open = open;
            this.Closed = closed;
            this.InProgress = false;
            this.Heuristic = heuristic;
            this.NodesPerFrame = 0; //by default here the "0" means we process all nodes in a single request, but you should change this
            this.TieBreakingWeight = tieBreakingWeight;
            this.pathfindingManager = GameObject.FindObjectOfType<PathfindingManager>();
        }
        public virtual void Preprocess()
        {
            //No preprocessing needed for basic A*
        }
        public virtual void InitializePathfindingSearch(int startX, int startY, int goalX, int goalY)
        {
            this.StartPositionX = startX;
            this.StartPositionY = startY;
            this.GoalPositionX = goalX;
            this.GoalPositionY = goalY;
            this.StartNode = gridGraph.GetNode(StartPositionX, StartPositionY);
            this.GoalNode = gridGraph.GetNode(GoalPositionX, GoalPositionY);

            //if it is not possible to quantize the positions and find the corresponding nodes, then we cannot proceed
            if (this.StartNode == null || this.GoalNode == null) return;

            // Reset debug and relevat variables here
            this.InProgress = true;
            this.TotalProcessedNodes = 0;
            this.TotalProcessingTime = 0.0f;
            this.MaxOpenNodes = 0;

            //Starting with the first node
            var initialNode = new NodeRecord(StartNode)
            {
                gCost = 0,
                hCost = CalculateHeuristic(this.StartNode),
            };

            //initialize open and closed lists
            initialNode.CalculateFCost(TieBreakingWeight);
            this.Open.Clear();
            this.Open.Add(initialNode);
            AddToOpenCalls++;
            this.Closed.Clear();
        }

        public virtual bool Search(out List<NodeRecord> solution, bool returnPartialSolution = false) {
            
            AStarPathfindingSearchCalls++;
            int ProcessedNodesPerFrame=0;
            NodeRecord closestNode=null;
            NodeRecord currentNode;

            //NodesPerFrame == 0 means there is no budget, so we search until we are done
            while (Open.CountOpen() > 0 && (NodesPerFrame == 0 || ProcessedNodesPerFrame < NodesPerFrame))
            {
                currentNode = Open.GetBestAndRemove();
                GetBestAndRemoveCalls++;

                //best node so far, used to build the partial path
                if (closestNode == null || currentNode.hCost < closestNode.hCost)
                {
                    closestNode = currentNode;
                }

                //we only test the goal when the node is expanded, otherwise A* is no longer optimal
                if (currentNode.Node.Equals(GoalNode))
                {
                    solution = CalculatePath(currentNode);
                    this.InProgress = false;
                    return true;
                }

                foreach (var connection in gridGraph.GetConnections(currentNode.Node))
                {
                    ProcessChildNode(currentNode, connection);
                }

                Closed.Add(currentNode);
                AddToClosedCalls++;
                currentNode.Node.status = VisualNodeStatus.Closed; //For visual purposes only

                TotalProcessedNodes++;
                ProcessedNodesPerFrame++;

                if (Open.CountOpen() > MaxOpenNodes) MaxOpenNodes = Open.CountOpen();
            }

            //the open set is empty, so there is no path to the goal
            if (Open.CountOpen() == 0)
            {
                solution = null;
                this.InProgress = false;
                return false;
            }

            //we ran out of budget for this frame, InProgress stays true so the manager calls us again
            solution = (returnPartialSolution && closestNode != null) ? CalculatePath(closestNode) : new List<NodeRecord>();
            return false;

    }
  
        protected virtual void ProcessChildNode(NodeRecord parentNode, Connection connection)
        {
            // Calculate newCost: parent cost + Calculate Distance Cont 
            // float newCost = parentNode.gCost + CalculateDistanceCost(parentNode, node) + this.Heuristic.H(node, this.GoalNode);

            Node node = connection.ToNode;
            NodeRecord newNodeRecord = new NodeRecord(node);
            float newCost = parentNode.gCost + connection.Cost;
            //float newCost = parentNode.gCost + CalculateDistanceCost(parentNode, node) + this.Heuristic.H(node, this.GoalNode);

            NodeRecord childFromClosed = Closed.Find(newNodeRecord);
            SearchInClosedCalls++;
            NodeRecord childFromOpen = Open.Find(newNodeRecord);
            SearchInOpenCalls++;

            
            //If node is not in any list ....
            if (childFromOpen == null && childFromClosed == null){

                newNodeRecord.parent=parentNode;
                newNodeRecord.gCost = newCost;
                newNodeRecord.hCost = CalculateHeuristic(node);
                newNodeRecord.CalculateFCost(TieBreakingWeight);
                Open.Add(newNodeRecord);
                node.status = VisualNodeStatus.Open; //For visual purposes only
                AddToOpenCalls++;
                
            
            }else{
               
                //If in Closed...
                
                if(childFromClosed != null && childFromClosed.gCost>newCost){
                    Closed.Remove(childFromClosed);
                    RemoveFromClosedCalls++;
                    newNodeRecord.parent = parentNode;
                    newNodeRecord.gCost = newCost;
                    newNodeRecord.hCost = CalculateHeuristic(node);
                    newNodeRecord.CalculateFCost(TieBreakingWeight);
                    Open.Add(newNodeRecord);
                    node.status = VisualNodeStatus.Open; //For visual purposes only
                    AddToOpenCalls++;
                }

                //If in Open..
                if (childFromOpen != null && childFromOpen.gCost>newCost){
                    newNodeRecord.parent = parentNode;
                    newNodeRecord.gCost = newCost;
                    newNodeRecord.hCost = CalculateHeuristic(node);
                    newNodeRecord.CalculateFCost(TieBreakingWeight);

                    Open.Replace(childFromOpen, newNodeRecord);
                    ReplaceCalls++;
                }
            }

            // Finally don't forget to update the actual Grid value: grid.SetGridObject(childNode.x, childNode.y, childNode);
            pathfindingManager.gridGraph.grid.SetGridObject(node.x, node.y, node);

        }

        protected virtual float CalculateHeuristic(Node node)
        {
            return this.Heuristic.H(node, this.GoalNode);
        }


        /*protected float CalculateDistanceCost(NodeRecord a, NodeRecord b)
        {
            // Math.abs is quite slow, thus we try to avoid it

            // ^ this is bs, Math.abs is much better than branching and is much more readable.
            // We could improve it by bit fiddling, but there is so much more to improve in this code than that obscure thing
            // https://stackoverflow.com/questions/6114099/fast-integer-abs-function

            int xDistance = Math.Abs(a.x - b.x);
            int yDistance = Math.Abs(a.y - b.y);
            int remaining = Math.Abs(xDistance - yDistance);

            // Diagonal Cost * Diagonal Size + Horizontal/Vertical Cost * Distance Left
            return MOVE_DIAGONAL_COST * Math.Min(xDistance, yDistance) + MOVE_STRAIGHT_COST * remaining;
        }*/

        // Method to calculate the Path, starts from the end Node and goes up until the beggining
        public List<NodeRecord> CalculatePath(NodeRecord endNode)
        {
            List<NodeRecord> path = new List<NodeRecord>();
            path.Add(endNode);

            // TODO implement
            // Start from the end node and go up until the beggining of the path
            NodeRecord nextNode = endNode.parent;

            while(nextNode!=null){
                path.Add(nextNode);
                nextNode = nextNode.parent;
            }
            path.Reverse();
            return path;
        }

    }
}
