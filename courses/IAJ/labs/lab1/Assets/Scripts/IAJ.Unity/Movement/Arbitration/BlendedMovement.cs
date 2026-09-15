using System.Collections.Generic;

namespace Assets.Scripts.IAJ.Unity.Movement.Arbitration
{
    public class MovementWithWeight
    {
        public DynamicMovement.DynamicMovement Movement { get; set; }
        public float Weight { get; set; }

        public MovementWithWeight(DynamicMovement.DynamicMovement movement)
        {
            this.Movement = movement;
            this.Weight = 1.0f;
        }

        public MovementWithWeight(DynamicMovement.DynamicMovement movement, float weight)
        {
            this.Movement = movement;
            this.Weight = weight;
        }
    }

    public class BlendedMovement : DynamicMovement.DynamicMovement
    {

        public override string Name
        {
            get
            {
                string description = "Blended Movement:\n";
                foreach (MovementWithWeight movementWithWeight in this.Movements)
                {
                    var active = movementWithWeight.Movement.Active;
                    var activeText = active ? " - active" : "";
                    description += movementWithWeight.Movement.Name + " (weight: " + movementWithWeight.Weight + ")" + activeText + "\n";
                }
                return description;
            }
        }

        public List<MovementWithWeight> Movements { get; private set; }

        public BlendedMovement()
        {
            this.Movements = new List<MovementWithWeight>();
            this.Output = new MovementOutput();
        }

        public override MovementOutput GetMovement()
        {
            MovementOutput tempOutput;

            this.Output.Clear();

            var totalWeight = 0.0f;

            foreach (MovementWithWeight movementWithWeight in this.Movements)
            {
                movementWithWeight.Movement.Character = this.Character;
                
                tempOutput = movementWithWeight.Movement.GetMovement();
                bool isActive = tempOutput.SquareMagnitude() > 0;
                movementWithWeight.Movement.Active = isActive;
                if (isActive)
                {
                    this.Output.linear += tempOutput.linear * movementWithWeight.Weight;
                    this.Output.angular += tempOutput.angular * movementWithWeight.Weight;
                    totalWeight += movementWithWeight.Weight;    
                }
            }

            if (totalWeight > 0)
            {
                //in case the totalWeight is not 0, we need to normalize
                float normalizationFactor = 1.0f/totalWeight;
                this.Output.linear *= normalizationFactor;
                this.Output.angular *= normalizationFactor;
            }

            return this.Output;
        }
    }
}
